"""Alsaify(사전학습) LOS(E1+E2) z-score 미적용 raw SDP cache 빌더 (per-lag ablation 용).

목적
----
기존 z-score Alsaify 캐시
  model/pretrained/checkpoints/dataset_cache_e12_w300_s300_lag1_20_tail_ps.npz
와 **동일 전처리 파라미터**로 RPCA->ACF->SDP 까지 수행하되, pipeline.py:104 의
per-window z-score 한 줄만 제외한 raw SDP 캐시를 산출한다. per-lag(D-020 B안) /
global(A안) 정규화는 이후 노트북에서 이 raw 캐시로부터 정규화만 교체해 일관되게
생성하므로, 이 빌더는 **raw 1종만** 저장한다(global z-score 캐시는 저장하지 않음).

설계 (origin/exp/perlag-cache:debug/modeling/build_rawsdp_cache.py 템플릿 재사용)
------------------------------------------------------------------------------
  - 데이터셋 무관 코어(재사용): 윈도우마다
      rpca_sparse(max_iter=200, tol=None) -> stacked_doppler_profile -> (28,20) raw SDP.
    pipeline.window_to_model_input 에서 line 104 의 per-window z-score만 제외한 것.
    pipeline.py 는 수정하지 않고 함수만 직접 호출한다(off 훅이 없으므로 직접 호출이 정석;
    analyze_sdp_energy.py:124-131 과 동일 패턴).
  - Alsaify 교체(front + 파일발견 + 라벨 + 파라미터):
      * 파일 발견  : train.py 의 실제 함수 _find_csvs(data_root, envs=LOS_ENVS) 재사용.
      * 라벨(6-cls): train.py 의 실제 매핑 상수 ACTIVITY_TO_LABEL 재사용(임의 매핑 금지).
                     클래스 순서/인덱스는 model.pretrained.model.CLASSES 규약 그대로.
      * front      : SafeSignal 경로가 아니라 Alsaify 경로 preprocess_file
                     (load_csi_csv->to_amplitude->downsample_alsaify->sliding_windows).
                     z-score 들어간 windows_to_model_input/preprocess_files_full 은
                     호출하지 않고, sliding_windows 윈도우에 코어를 직접 적용.
  - 저장: .npz.tmp 로 쓰고 sanity 통과 후 os.replace. normalization="none_raw_sdp".

실행
----
  # dry-run 만 (기본). 신규 파일 추가 후 검증용 — 본 빌드 안 함.
  python debug/modeling/build_alsaify_rawsdp_cache.py

  # 본 빌드(전체) — Alsaify 원본(E1+E2)을 가진 팀원 PC 에서:
  python debug/modeling/build_alsaify_rawsdp_cache.py --full --workers 18

주의(Windows/spawn): ProcessPoolExecutor 사용 → 반드시 __main__ 진입점에서 실행.
모듈 최상위는 torch 등 무거운 import 를 피한다(워커 spawn 비용). train.py/model import 는
main() 안에서만(워커는 코어 함수만 필요).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# 데이터셋 무관 코어 + Alsaify front (torch 미포함 — 워커 spawn 시에도 가벼움)
from model.preprocessing.pipeline import preprocess_file
from model.preprocessing.rpca import rpca_sparse
from model.preprocessing.sdp import stacked_doppler_profile, W_T
from model.preprocessing.acf import N_LAGS

# ── 경로 ─────────────────────────────────────────────────────────────────────
DATA_ROOT = PROJECT_ROOT / "data" / "alsaify-raw"
CKPT_DIR = PROJECT_ROOT / "model" / "pretrained" / "checkpoints"
EXISTING_ZSCORE_CACHE = CKPT_DIR / "dataset_cache_e12_w300_s300_lag1_20_tail_ps.npz"
OUT_FINAL = CKPT_DIR / "dataset_cache_e12_w300_s300_lag1_20_tail_ps_rawsdp.npz"
OUT_TMP = Path(str(OUT_FINAL) + ".tmp")
LOG_DIR = PROJECT_ROOT / "debug" / "modeling" / "diag_out"

# ── 빌드 파라미터 (기존 Alsaify z-score 캐시와 반드시 동일) ──────────────────
# 값 자체는 train.py 의 CACHE_* 상수에서 main() 에서 import 해 검증/출력한다.
# 주의: SafeSignal raw 빌드는 pad_short=False 였으나 Alsaify 는 pad_short=True.
PRE_KW = dict(
    window_size=300,   # CACHE_WINDOW_SIZE (=WINDOW_SIZE, 3s@100Hz)
    stride=None,       # CACHE_STRIDE (None -> window_size, 비중첩)
    drop_last=True,    # preprocess_files_full 기본값과 동일
    tail_window=True,  # CACHE_TAIL_WINDOW
    pad_short=True,    # CACHE_PAD_SHORT  ← Alsaify 는 True
)
RPCA_MAX_ITER = 200
RPCA_TOL = None
LAG_POLICY = "lag1_20"  # ACF lag0 제외, N_LAGS=20

# SetThreadExecutionState flags (절전 억제)
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


# ── Tee 로그 (cp949 인코딩 예외 처리 포함) ───────────────────────────────────
class Tee:
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


def suppress_sleep() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log("[sleep] SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED) 적용 (절전 억제)")
    except Exception as e:  # noqa: BLE001
        log(f"[sleep] 억제 실패(무시하고 진행): {e!r}")


def restore_sleep() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:  # noqa: BLE001
        pass


# ── 데이터셋 무관 코어 (build_rawsdp_cache.raw_sdp_for_window 패턴 그대로) ────
def raw_sdp_for_window(window: np.ndarray) -> np.ndarray:
    """단일 윈도우 (300, n_sc) -> raw SDP (W_T, N_LAGS). z-score 미적용."""
    sparse = rpca_sparse(window, max_iter=RPCA_MAX_ITER, tol=RPCA_TOL)
    return stacked_doppler_profile(sparse)


def zscore_window(sdp: np.ndarray) -> np.ndarray:
    """pipeline.py:104 와 동일한 per-window global z-score (검증용 — 저장 안 함)."""
    return (sdp - sdp.mean()) / (sdp.std() + 1e-6)


# ── ProcessPoolExecutor 워커 (모듈 최상위, picklable) ────────────────────────
def _raw_worker(path_str: str):
    """단일 Alsaify CSV -> raw SDP 윈도우 배열 + 메타.

    Returns (basename, arr(N,1,W_T,N_LAGS)|None, activity:int|None,
             subject:int|None, env:int|None, error|None).
    """
    try:
        pre = preprocess_file(path_str, **PRE_KW)  # Alsaify front (downsample_alsaify)
        w = pre.windows  # (n, 300, n_sc)
        if w.shape[0] == 0:
            return (os.path.basename(path_str), None, None, None, None, "empty_windows")
        out = np.empty((w.shape[0], 1, W_T, N_LAGS), dtype=np.float32)
        for i, win in enumerate(w):
            out[i, 0] = raw_sdp_for_window(win)
        m = pre.meta
        return (os.path.basename(path_str), out, int(m.activity), int(m.subject), int(m.environment), None)
    except Exception as e:  # noqa: BLE001
        return (os.path.basename(path_str), None, None, None, None, repr(e))


# ── 검증 헬퍼 ────────────────────────────────────────────────────────────────
def _per_window_zscore_stats(X: np.ndarray) -> dict:
    """X (N,1,28,20) raw -> 각 윈도우에 global z-score 적용 후 mean/std 통계."""
    n = X.shape[0]
    means = np.empty(n, dtype=np.float64)
    stds = np.empty(n, dtype=np.float64)
    raw_stds = np.empty(n, dtype=np.float64)
    for i in range(n):
        sdp = X[i, 0]
        raw_stds[i] = float(sdp.std())
        z = zscore_window(sdp)
        means[i] = float(z.mean())
        stds[i] = float(z.std())
    return {
        "n": n,
        "raw_std_min": float(raw_stds.min()),
        "raw_std_med": float(np.median(raw_stds)),
        "raw_std_max": float(raw_stds.max()),
        "z_mean_absmax": float(np.abs(means).max()),
        "z_std_min": float(stds.min()),
        "z_std_med": float(np.median(stds)),
        "z_std_max": float(stds.max()),
    }


def _print_class_counts(y: np.ndarray, classes: list[str], tag: str) -> dict:
    counts = {name: int((y == i).sum()) for i, name in enumerate(classes)}
    log(f"  [{tag}] windows={int(len(y))}  class_counts={counts}")
    return counts


def _distribution_sanity(classes: list[str]) -> None:
    """기존 z-score Alsaify 캐시 집계(윈도우 수/클래스/ std) 출력 — 비교 기준 제공.

    행 단위 allclose 는 비결정 순서(train.py 병렬 완료순)로 불가하므로 시도하지 않고
    집계만 비교한다.
    """
    log("\n[dist-sanity] 기존 z-score Alsaify 캐시(비교 기준):")
    if not EXISTING_ZSCORE_CACHE.exists():
        log(f"  (없음 — 로컬에 {EXISTING_ZSCORE_CACHE.name} 미존재. 팀원 PC 에서 비교)")
        return
    z = np.load(EXISTING_ZSCORE_CACHE, allow_pickle=True)
    X, y = z["X"], z["y"]
    log(f"  file={EXISTING_ZSCORE_CACHE.name}")
    log(f"  X.shape={tuple(X.shape)}  X.std={float(X.std()):.5f} (z-score 캐시이므로 ≈1 기대)")
    _print_class_counts(y, classes, "기존(z-score)")
    log("  → 새 raw 빌드의 윈도우 수/클래스 분포가 위와 같은 자릿수/규모(6-class)인지 비교할 것"
        " (행 allclose 는 비결정 순서로 불가, 집계만 비교)")


def _save_raw_cache(X, y, subject, env, filename, classes) -> None:
    """raw 1종만 저장 (.tmp -> os.replace). normalization='none_raw_sdp'."""
    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TMP, "wb") as fh:
        np.savez_compressed(
            fh,
            X=X.astype(np.float32, copy=False),
            y=np.asarray(y, dtype=np.int64),
            subject=np.asarray(subject, dtype=np.int64),
            env=np.asarray(env, dtype=np.int64),
            filename=np.asarray(filename, dtype=object),
            classes=np.asarray(classes, dtype=object),
            normalization="none_raw_sdp",
            lag_policy=LAG_POLICY,
        )
    os.replace(OUT_TMP, OUT_FINAL)
    log(f"[write] {OUT_FINAL} ({OUT_FINAL.stat().st_size/1e6:.1f} MB)")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Alsaify raw SDP cache 빌더 (per-lag ablation)")
    ap.add_argument("--full", action="store_true",
                    help="본 빌드(전체 실행 + 저장). 미지정 시 dry-run 만 수행.")
    ap.add_argument("--workers", type=int, default=18, help="본 빌드 병렬 워커 수")
    ap.add_argument("--dry-files", type=int, default=5, help="dry-run 처리 파일 수")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(LOG_DIR / "alsaify_rawsdp_build.log")

    # 실제 함수/상수 import (코드에서 확인된 이름만). 워커가 아닌 main 에서만.
    from model.pretrained.train import (
        _find_csvs,
        ACTIVITY_TO_LABEL,
        LOS_ENVS,
        CACHE_WINDOW_SIZE,
        CACHE_STRIDE,
        CACHE_TAIL_WINDOW,
        CACHE_PAD_SHORT,
    )
    from model.pretrained.model import CLASSES

    classes = list(CLASSES)
    t0 = time.time()
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    log("=" * 70)
    log(f"[start] {start}  Alsaify raw SDP cache build  (mode={'FULL' if args.full else 'DRY-RUN'})")
    suppress_sleep()

    # ── 재사용한 실제 이름 + 파라미터 정합성 출력 ────────────────────────────
    log("[reuse] 파일발견=model.pretrained.train._find_csvs  "
        "파싱=model.preprocessing.loader.parse_alsaify_filename  "
        "라벨=model.pretrained.train.ACTIVITY_TO_LABEL  front=pipeline.preprocess_file")
    log(f"[classes] {classes}  (인덱스 0..{len(classes)-1})")
    # train.py CACHE_* 상수와 본 빌더 PRE_KW 정합성 검증 (불일치 시 즉시 멈춤)
    expect = {
        "window_size": CACHE_WINDOW_SIZE,
        "stride": CACHE_STRIDE,
        "tail_window": CACHE_TAIL_WINDOW,
        "pad_short": CACHE_PAD_SHORT,
    }
    mismatch = {k: (PRE_KW.get(k), v) for k, v in expect.items() if PRE_KW.get(k) != v}
    if mismatch:
        log(f"[FAIL] PRE_KW 가 train.py CACHE_* 와 불일치: {mismatch}")
        restore_sleep()
        return 1
    log(f"[params] envs={tuple(LOS_ENVS)} window_size={PRE_KW['window_size']} "
        f"stride={PRE_KW['stride']}(->비중첩) lag_policy={LAG_POLICY}(N_LAGS={N_LAGS}) "
        f"tail_window={PRE_KW['tail_window']} pad_short={PRE_KW['pad_short']} "
        f"rpca_max_iter={RPCA_MAX_ITER} rpca_tol={RPCA_TOL}  (W_T={W_T})")

    # ── 파일 발견 (train.py 실제 로직) ───────────────────────────────────────
    csvs = _find_csvs(DATA_ROOT, envs=tuple(LOS_ENVS))
    log(f"[discover] _find_csvs(envs={tuple(LOS_ENVS)}) -> {len(csvs)} CSV "
        f"(매핑 가능 활동만; data_root={DATA_ROOT})")
    if not csvs:
        log("[FAIL] 매핑 가능한 Alsaify CSV 0개. data/alsaify-raw/Environment {1,2} 압축 해제 확인.")
        restore_sleep()
        return 1

    # ── DRY-RUN (앞 N개 파일, 단일 프로세스) ─────────────────────────────────
    n_dry = min(args.dry_files, len(csvs))
    log(f"\n[dry-run] 앞 {n_dry}개 파일 처리 (단일 프로세스)")
    dry_blocks, dry_y, dry_sub, dry_env = [], [], [], []
    for p in csvs[:n_dry]:
        bn, arr, act, sub, env, err = _raw_worker(str(p))
        if err is not None:
            log(f"[FAIL] dry-run 파일 처리 실패: {bn}: {err}")
            restore_sleep()
            return 1
        if act not in ACTIVITY_TO_LABEL:
            log(f"[FAIL] dry-run 활동코드 매핑 불가(A{act}): {bn}")
            restore_sleep()
            return 1
        label = ACTIVITY_TO_LABEL[act]
        dry_blocks.append(arr)
        dry_y.extend([label] * arr.shape[0])
        dry_sub.extend([sub] * arr.shape[0])
        dry_env.extend([env] * arr.shape[0])
        log(f"  {bn}: windows={arr.shape[0]} A{act}->y={label}({classes[label]}) S{sub} E{env}")
    dry_X = np.concatenate(dry_blocks, axis=0)
    dry_y_arr = np.asarray(dry_y, dtype=np.int64)

    # (i) raw std 가 1이 아닌 0.0x 대인지
    log(f"\n[dry-run/raw] X_raw.shape={tuple(dry_X.shape)} dtype={dry_X.dtype} "
        f"전체 std={float(dry_X.std()):.5f} min={float(dry_X.min()):.5f} max={float(dry_X.max()):.5f}")
    # (ii) self-consistent: 윈도우별 global z-score -> mean≈0, std≈1
    st = _per_window_zscore_stats(dry_X)
    log(f"[dry-run/self-consistent] per-window raw std: "
        f"min={st['raw_std_min']:.5f} med={st['raw_std_med']:.5f} max={st['raw_std_max']:.5f}  "
        f"(1 아님 = raw 특성 OK)")
    log(f"[dry-run/self-consistent] global z-score(raw) 후: "
        f"|mean|max={st['z_mean_absmax']:.2e}  std[min/med/max]="
        f"{st['z_std_min']:.5f}/{st['z_std_med']:.5f}/{st['z_std_max']:.5f}  (mean≈0, std≈1 기대)")
    mean_ok = st["z_mean_absmax"] < args.atol
    std_ok = abs(st["z_std_med"] - 1.0) < 1e-2
    raw_ok = st["raw_std_med"] < 0.5  # raw SDP 는 0.0x 대(1 아님)
    log(f"[dry-run/verdict] mean≈0:{mean_ok}  std_med≈1:{std_ok}  raw_std<0.5:{raw_ok}  "
        f"=> {'PASS' if (mean_ok and std_ok and raw_ok) else 'CHECK'}")
    _print_class_counts(dry_y_arr, classes, "dry-run")

    # ── 분포 sanity (기존 z-score 캐시 집계 비교 기준) ───────────────────────
    _distribution_sanity(classes)

    if not args.full:
        log("\n[dry-run only] 본 빌드(전체)는 생략했습니다. (raw 캐시 미저장)")
        log("[next] 팀원이 본 빌드를 돌릴 실행 명령(아래 한 줄):")
        log(f"  python debug/modeling/build_alsaify_rawsdp_cache.py --full --workers {args.workers}")
        log(f"  (출력: {OUT_FINAL}  — gitignore 대상, 커밋 안 함)")
        restore_sleep()
        log(f"[end] dry-run 완료 [{time.time()-t0:.1f}s]")
        return 0

    # ── 본 빌드 (병렬) — 팀원 PC 전용 ────────────────────────────────────────
    log(f"\n[full] 본 빌드 시작: files={len(csvs)} workers={args.workers}")
    results: dict[str, tuple] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_raw_worker, str(p)) for p in csvs]
        for fut in as_completed(futures):
            bn, arr, act, sub, env, err = fut.result()
            if err is not None:
                log(f"[FAIL] 본 빌드 파일 처리 실패: {bn}: {err}")
                restore_sleep()
                return 1
            results[bn] = (arr, act, sub, env)
            done += 1
            if done % 100 == 0:
                log(f"[progress] {done}/{len(csvs)} files")

    # csvs 순서대로 결정적 조립 (train.py 의 완료순과 달리 재현 가능 순서)
    Xs, ys, subs, envs_l, fnames = [], [], [], [], []
    for p in csvs:
        bn = p.name
        if bn not in results:
            log(f"[FAIL] 결과 누락: {bn}")
            restore_sleep()
            return 1
        arr, act, sub, env = results[bn]
        if act not in ACTIVITY_TO_LABEL:
            log(f"[FAIL] 활동코드 매핑 불가(A{act}): {bn}")
            restore_sleep()
            return 1
        label = ACTIVITY_TO_LABEL[act]
        Xs.append(arr)
        ys.extend([label] * arr.shape[0])
        subs.extend([sub] * arr.shape[0])
        envs_l.extend([env] * arr.shape[0])
        fnames.extend([bn] * arr.shape[0])
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)

    # sanity: shape / 6-class
    if X.shape[1:] != (1, W_T, N_LAGS):
        log(f"[FAIL] X shape 비정상: {tuple(X.shape)}")
        restore_sleep()
        return 1
    if not (y.min() >= 0 and y.max() <= len(classes) - 1):
        log(f"[FAIL] y 라벨 범위 이탈: {np.unique(y)}")
        restore_sleep()
        return 1
    full_st = _per_window_zscore_stats(X[: min(200, X.shape[0])])
    log(f"[full/self-consistent] (앞 200 윈도우) |mean|max={full_st['z_mean_absmax']:.2e} "
        f"std_med={full_st['z_std_med']:.5f} raw_std_med={full_st['raw_std_med']:.5f}")
    log(f"[full/assemble] X={tuple(X.shape)}")
    _print_class_counts(y, classes, "raw(full)")
    _distribution_sanity(classes)

    _save_raw_cache(X, y, subs, envs_l, fnames, classes)
    restore_sleep()
    log(f"[end] 본 빌드 완료 [{(time.time()-t0)/60:.1f}min]  out={OUT_FINAL.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
