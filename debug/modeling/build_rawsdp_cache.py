"""finetune7 cache의 filename 목록으로 z-score 미적용 raw SDP cache를 빌드한다.

목적: per-lag 정규화는 다음 학습 단계에서 train split 통계로 fit하기 위해,
누수 방지를 위해 정규화 이전(raw) SDP를 그대로 저장한다.

핵심 설계 (사람 승인 사항):
  - 입력 소스 = safesignal_e1234_finetune7.npz 의 filename 목록(1439개). 디렉터리
    glob을 쓰지 않고, cache에 기록된 정확한 파일 세트/순서를 그대로 재현한다.
  - 각 파일: preprocess_safesignal_file(load->resample->sliding_windows,
    window300/stride None/tail_window True/rx both) -> 윈도우별 rpca_sparse(max_iter200,
    tol None) -> stacked_doppler_profile -> raw SDP(28,20).
    => pipeline.window_to_model_input 에서 line 104의 per-window z-score만 제외한 것.
    pipeline.py는 수정하지 않고 함수만 직접 호출한다.
  - 윈도우 순서/개수가 finetune7(3581 windows)과 정확히 일치하도록 cache 순서대로 조립.
  - DRY-RUN: 앞 5개 파일 raw SDP에 (x-mean)/(std+1e-6) 적용 -> cache X 해당 행과 allclose.
    불통과 시 _FAILED.flag 후 중단.
  - 전역/per-lag 통계 미적용 (raw 그대로).

산출: safesignal_e1234_finetune7_rawsdp.npz (+ .summary.json). .npz.tmp 로 쓰고
sanity 통과 후에만 rename. D:\handoff 로 복사 + DONE/FAILED flag.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model.preprocessing.pipeline import preprocess_safesignal_file
from model.preprocessing.rpca import rpca_sparse
from model.preprocessing.sdp import stacked_doppler_profile, W_T
from model.preprocessing.acf import N_LAGS

CACHE_IN = PROJECT_ROOT / "model" / "finetune" / "cache" / "safesignal_e1234_finetune7.npz"
SUMMARY_IN = CACHE_IN.with_suffix(".summary.json")
OUT_FINAL = PROJECT_ROOT / "model" / "finetune" / "cache" / "safesignal_e1234_finetune7_rawsdp.npz"
OUT_TMP = Path(str(OUT_FINAL) + ".tmp")
SUMMARY_OUT = OUT_FINAL.with_suffix(".summary.json")
DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
LOG_DIR = PROJECT_ROOT / "debug" / "modeling" / "diag_out"
HANDOFF = Path(r"D:\handoff")

# finetune7 빌드와 동일한 전처리 파라미터 (build_safesignal_cache.py 기준)
PRE_KW = dict(
    rx="both",
    target_hz=100.0,
    max_gap_ms=100.0,
    window_size=300,
    stride=None,
    drop_last=True,
    tail_window=True,
    pad_short=False,
)
RPCA_MAX_ITER = 200
RPCA_TOL = None

# SetThreadExecutionState flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class Tee:
    """stdout과 로그 파일에 동시 출력."""

    def __init__(self, log_path: Path):
        self._console = sys.__stdout__
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, s):
        try:
            self._console.write(s)
        except UnicodeEncodeError:
            enc = getattr(self._console, "encoding", None) or "utf-8"
            self._console.write(s.encode(enc, errors="replace").decode(enc, errors="replace"))
        self._file.write(s)

    def flush(self):
        self._console.flush()
        self._file.flush()


def log(msg: str) -> None:
    print(msg, flush=True)


def mem_used_mb() -> float:
    """시스템 사용 메모리(MB) 대략값 (psutil 없이 ctypes)."""
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return (st.ullTotalPhys - st.ullAvailPhys) / (1024 * 1024)


def raw_sdp_for_window(window: np.ndarray) -> np.ndarray:
    """단일 윈도우 (300, n_sc) -> raw SDP (W_T, N_LAGS). z-score 미적용."""
    sparse = rpca_sparse(window, max_iter=RPCA_MAX_ITER, tol=RPCA_TOL)
    return stacked_doppler_profile(sparse)


def _raw_worker(path_str: str):
    """ProcessPoolExecutor 워커. 모듈 최상위 (picklable).

    Returns (basename, raw_array(N,1,W_T,N_LAGS) | None, error | None).
    """
    try:
        pre = preprocess_safesignal_file(path_str, **PRE_KW)
        w = pre.windows  # (n, 300, n_sc)
        if w.shape[0] == 0:
            return (os.path.basename(path_str), None, "empty_windows")
        out = np.empty((w.shape[0], 1, W_T, N_LAGS), dtype=np.float32)
        for i, win in enumerate(w):
            out[i, 0] = raw_sdp_for_window(win)
        return (os.path.basename(path_str), out, None)
    except Exception as e:  # noqa: BLE001
        return (os.path.basename(path_str), None, repr(e))


def zscore_window(sdp: np.ndarray) -> np.ndarray:
    """pipeline.py:104 와 동일한 per-window z-score."""
    return (sdp - sdp.mean()) / (sdp.std() + 1e-6)


def fail(reason: str, extra: dict | None = None) -> None:
    """_FAILED.flag 작성 후 종료."""
    HANDOFF.mkdir(parents=True, exist_ok=True)
    payload = {"status": "FAILED", "reason": reason, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        payload.update(extra)
    (HANDOFF / "perlag_cache_FAILED.flag").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"[FAILED] {reason} | extra={extra}")
    sys.exit(1)


def build_order(cache):
    """cache filename 배열에서 (basename, count) 를 등장(연속 블록) 순서로 추출.

    각 파일이 단일 연속 블록인지도 검증 (interleave 없어야 순서 재현 가능).
    """
    fn = cache["filename"].astype(str)
    order = []  # list[(name, count)]
    runs = OrderedDict()
    i = 0
    n = len(fn)
    seen_names = set()
    while i < n:
        name = fn[i]
        j = i
        while j < n and fn[j] == name:
            j += 1
        cnt = j - i
        if name in seen_names:
            fail("cache filename interleaved (단일 연속 블록 아님)", {"file": name})
        seen_names.add(name)
        order.append((name, cnt))
        runs[name] = cnt
        i = j
    return order


def resolve_paths(order):
    """basename -> 디스크 경로 (data/cleaned 재귀). 누락 시 FAILED."""
    name_to_path = {}
    for p in DATA_DIR.rglob("*.csv"):
        name_to_path.setdefault(p.name, p)
    missing = [name for name, _ in order if name not in name_to_path]
    if missing:
        fail("cache 파일을 디스크에서 못 찾음", {"missing_count": len(missing), "examples": missing[:5]})
    return name_to_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--dry-files", type=int, default=5)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(LOG_DIR / "perlag_cache_build.log")

    start = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    log("=" * 70)
    log(f"[start] {start}  raw SDP cache build (per-lag 용)")

    # sleep 억제
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log("[sleep] SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED) 적용 (절전 억제)")
    except Exception as e:  # noqa: BLE001
        log(f"[sleep] 억제 실패(무시하고 진행): {e!r}")

    if not CACHE_IN.exists():
        fail(f"입력 cache 없음: {CACHE_IN}")
    cache = np.load(CACHE_IN, allow_pickle=True)
    X_ref = cache["X"]
    n_total = X_ref.shape[0]
    log(f"[input] finetune7 cache: X={X_ref.shape} windows={n_total}")

    order = build_order(cache)
    log(f"[input] distinct files={len(order)}  (sum windows={sum(c for _, c in order)})")
    if sum(c for _, c in order) != n_total:
        fail("filename 블록 합 != X 행수")

    name_to_path = resolve_paths(order)

    # ── DRY-RUN ────────────────────────────────────────────────────────────
    n_dry = min(args.dry_files, len(order))
    dry_files = order[:n_dry]
    dry_window_total = sum(c for _, c in dry_files)
    log(f"[dry-run] {n_dry} files, expected windows={dry_window_total}")
    td0 = time.time()
    dry_raw = []
    for name, cnt in dry_files:
        bn, arr, err = _raw_worker(str(name_to_path[name]))
        if err is not None:
            fail("dry-run 파일 처리 실패", {"file": name, "error": err})
        if arr.shape[0] != cnt:
            fail("dry-run window 수 불일치", {"file": name, "got": int(arr.shape[0]), "expected": int(cnt)})
        dry_raw.append(arr)
    dry_per_file = (time.time() - td0) / max(n_dry, 1)
    dry_X_raw = np.concatenate(dry_raw, axis=0)
    dry_z = np.empty_like(dry_X_raw)
    for i in range(dry_X_raw.shape[0]):
        dry_z[i, 0] = zscore_window(dry_X_raw[i, 0])
    ref_slice = X_ref[:dry_window_total]
    max_abs = float(np.max(np.abs(dry_z - ref_slice)))
    ok = np.allclose(dry_z, ref_slice, atol=args.atol, rtol=args.rtol)
    log(f"[dry-run] max_abs_diff={max_abs:.3e}  allclose(atol={args.atol},rtol={args.rtol})={ok}")
    if not ok:
        fail(
            "DRY-RUN allclose 불통과 (raw->zscore != finetune7 X)",
            {"max_abs_diff": max_abs, "shape": list(dry_z.shape), "files": [n for n, _ in dry_files]},
        )
    log("[dry-run] 통과 → 본실행 진행")

    # 예상 소요시간 (병렬 추정)
    est_sec = dry_per_file * len(order) / max(args.workers, 1)
    log(
        f"[plan] total windows={n_total} files={len(order)} workers={args.workers} "
        f"dry_per_file={dry_per_file:.2f}s 예상≈{est_sec/60:.1f}min mem_used={mem_used_mb():.0f}MB"
    )

    # ── 본실행 (병렬) ────────────────────────────────────────────────────────
    paths = [str(name_to_path[name]) for name, _ in order]
    results: dict[str, np.ndarray] = {}
    mem_peak = mem_used_mb()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_raw_worker, p) for p in paths]
        for fut in as_completed(futures):
            bn, arr, err = fut.result()
            if err is not None:
                fail("본실행 파일 처리 실패", {"file": bn, "error": err})
            results[bn] = arr
            done += 1
            if done % 100 == 0:
                m = mem_used_mb()
                mem_peak = max(mem_peak, m)
                log(f"[progress] {done}/{len(order)} files  mem_used={m:.0f}MB")
    mem_peak = max(mem_peak, mem_used_mb())

    # cache 순서대로 조립 + provenance
    blocks = []
    within_index = []
    for name, cnt in order:
        arr = results.get(name)
        if arr is None:
            fail("결과 누락", {"file": name})
        if arr.shape[0] != cnt:
            fail("본실행 window 수 불일치", {"file": name, "got": int(arr.shape[0]), "expected": int(cnt)})
        blocks.append(arr)
        within_index.extend(range(cnt))
    X_raw = np.concatenate(blocks, axis=0).astype(np.float32)
    within_arr = np.asarray(within_index, dtype=np.int64)

    log(f"[assemble] X_raw={X_raw.shape} mem_peak≈{mem_peak:.0f}MB workers={args.workers}")

    # ── sanity ───────────────────────────────────────────────────────────────
    if X_raw.shape != (n_total, 1, W_T, N_LAGS):
        fail("X_raw shape 불일치", {"got": list(X_raw.shape), "expected": [n_total, 1, W_T, N_LAGS]})
    if X_raw.shape[0] != 3581:
        fail("window 수 3581 불일치", {"got": int(X_raw.shape[0])})
    # 클래스 분포 일치 (y는 cache에서 그대로 승계)
    y = cache["y"]
    classes = cache["classes"].astype(str).tolist()
    class_counts = {name: int((y == i).sum()) for i, name in enumerate(classes)}
    summary_ref = json.loads(SUMMARY_IN.read_text(encoding="utf-8")) if SUMMARY_IN.exists() else None
    if summary_ref and class_counts != summary_ref.get("class_counts"):
        fail("클래스 분포 불일치", {"got": class_counts, "ref": summary_ref.get("class_counts")})
    # 강한 정합성: 전체 zscore(X_raw) allclose finetune7 X
    full_z = np.empty_like(X_raw)
    for i in range(X_raw.shape[0]):
        full_z[i, 0] = zscore_window(X_raw[i, 0])
    full_max_abs = float(np.max(np.abs(full_z - X_ref)))
    full_ok = np.allclose(full_z, X_ref, atol=args.atol, rtol=args.rtol)
    log(f"[sanity] full zscore(X_raw) vs finetune7 X: max_abs_diff={full_max_abs:.3e} allclose={full_ok}")
    if not full_ok:
        fail("full allclose 불통과", {"max_abs_diff": full_max_abs})
    log(f"[sanity] window={X_raw.shape[0]} shape OK, class_counts 일치, full allclose OK")

    # ── 저장 (.tmp -> rename) ─────────────────────────────────────────────────
    with open(OUT_TMP, "wb") as fh:
        np.savez_compressed(
            fh,
            X=X_raw,
            y=y,
            subject=cache["subject"],
            source=cache["source"],
            env=cache["env"],
            filename=cache["filename"],
            activity=cache["activity"],
            trial=cache["trial"],
            within_file_index=within_arr,
            classes=cache["classes"],
            class_policy=str(cache["class_policy"]),
            is_augmented=cache["is_augmented"],
            normalization="none_raw_sdp",
        )
    os.replace(OUT_TMP, OUT_FINAL)
    log(f"[write] {OUT_FINAL} ({OUT_FINAL.stat().st_size/1e6:.1f} MB)")

    end = time.strftime("%Y-%m-%d %H:%M:%S")
    summary = {
        "source": "finetune7 cache filename 목록 기반 (디렉터리 glob 미사용)",
        "input_cache": str(CACHE_IN),
        "policy": str(cache["class_policy"]),
        "classes": classes,
        "windows": int(X_raw.shape[0]),
        "files": len(order),
        "shape": list(X_raw.shape),
        "normalization": "none (raw SDP, pipeline.py:104 z-score 미적용)",
        "preprocess": {**PRE_KW, "rpca_max_iter": RPCA_MAX_ITER, "rpca_tol": RPCA_TOL},
        "class_counts": class_counts,
        "dry_run_max_abs_diff": max_abs,
        "full_zscore_max_abs_diff": full_max_abs,
        "workers": args.workers,
        "mem_peak_mb": round(mem_peak),
        "start": start,
        "end": end,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[write] {SUMMARY_OUT}")

    # ── handoff 복사 + DONE flag ──────────────────────────────────────────────
    HANDOFF.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(OUT_FINAL, HANDOFF / OUT_FINAL.name)
    shutil.copy2(SUMMARY_OUT, HANDOFF / SUMMARY_OUT.name)
    shutil.copy2(LOG_DIR / "perlag_cache_build.log", HANDOFF / "perlag_cache_build.log")
    (HANDOFF / "perlag_cache_DONE.flag").write_text(
        json.dumps(
            {
                "status": "DONE",
                "start": start,
                "end": end,
                "windows": int(X_raw.shape[0]),
                "files": len(order),
                "out": str(OUT_FINAL),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log(f"[done] {end}  windows={X_raw.shape[0]} -> handoff 복사 + DONE.flag")

    # sleep 억제 해제
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
