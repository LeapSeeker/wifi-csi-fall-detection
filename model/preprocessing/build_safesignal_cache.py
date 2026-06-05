"""SafeSignal 자체수집 데이터 → NPZ 캐시 생성.

모드 A (기본): stride 기반 윈도우 — fall_stride=50, max_start=100
모드 B (어노테이션): fall_onsets.json 참조 → onset 중심 윈도우 2개
  [onset-100 : onset+200]  — 낙상 피크 정중앙
  [onset-50  : onset+250]  — 50패킷 뒤 시프트

사용법:
    # 모드 A (stride 기반)
    python -m model.preprocessing.build_safesignal_cache \\
        --src dummy_src \\
        --out model/finetune/cache/ss_stride50_p6.npz

    # 모드 B (어노테이션 기반, 권장)
    python -m model.preprocessing.build_safesignal_cache \\
        --src dummy_src \\
        --out model/finetune/cache/ss_annotated_p6.npz \\
        --annotations dummy_src/fall_onsets.json
"""
from __future__ import annotations

# BLAS 스레드 1로 고정 — 워커 N개 × BLAS M스레드 = N×M 코어 경합 방지.
# numpy import 전에 설정해야 적용됨.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.preprocessing.loader import (
    load_safesignal_csv, parse_safesignal_filename,
)
from model.preprocessing.pipeline import (
    preprocess_safesignal_file_full, windows_to_model_input,
)
from model.preprocessing.resample import resample_to_100hz
from model.preprocessing.window import WINDOW_SIZE

# ── 수집 구조 상수 (100 Hz 기준) ──────────────────────────────────────
FALL_ONSET = 100   # 어노테이션 없을 때 기본 onset 가정값
FALL_MAX_START = FALL_ONSET

# 어노테이션 기반 윈도우: onset 기준 앞뒤 오프셋
# [onset + shift : onset + shift + 300], shift ∈ ANNOTATED_SHIFTS
ANNOTATED_SHIFTS = (-100, -50, 0)   # 세션당 3 윈도우 (stride 기반과 동일 수량)

# ── 클래스 매핑 (pretrained6) ─────────────────────────────────────────
CLASS_NAMES = ("fall", "walking", "sit_stand", "lying", "standing", "picking")

ACTIVITY_TO_LABEL: dict[str, int] = {
    "FALL_SIT_B": 0, "FALL_SIT_F": 0,
    "FALL_STD_B": 0, "FALL_STD_F": 0,
    "FALL_WALK_B": 0, "FALL_WALK_F": 0,
    "WALK":    1,
    "SIT_STD": 2,
    "LIE":     3,
    "STAND":   4,
    "PICK":    5,
    # RUN → pretrained6 미포함, 스킵
}

SOURCE = "safesignal"
CLASS_POLICY = "pretrained6"
NORMALIZATION = "global_self"


# ── 워커 (module-level: ProcessPoolExecutor pickling 필수) ────────────

def _extract_annotated_fall(
    amp_100hz: np.ndarray,
    onset: int,
    window_size: int,
) -> np.ndarray:
    """onset 기준으로 윈도우 2개 추출 → (2, window_size, n_sc)."""
    n = len(amp_100hz)
    windows = []
    for shift in ANNOTATED_SHIFTS:
        start = onset + shift
        end   = start + window_size
        if start < 0:
            start, end = 0, window_size
        if end > n:
            start, end = n - window_size, n
        if start < 0:
            continue
        windows.append(amp_100hz[start:end])
    if not windows:
        return np.empty((0, window_size, amp_100hz.shape[1]), dtype=np.float32)
    return np.stack(windows).astype(np.float32)


def _worker(
    args: tuple[str, int, int, int, int],
) -> tuple[str, list[dict] | None, str | None]:
    """파일 하나를 처리해 레코드 리스트를 반환.

    args: (path, fall_stride, nonfail_stride, window_size, onset)
      onset = -1 이면 stride 기반 모드 (모드 A)
      onset >= 0 이면 어노테이션 기반 모드 (모드 B)
    """
    path, fall_stride, nonfail_stride, window_size, onset = args
    try:
        meta = parse_safesignal_filename(path)
        activity = meta.activity.upper()

        if activity not in ACTIVITY_TO_LABEL:
            return path, None, f"skip:{activity}"

        is_fall = activity.startswith("FALL")
        label   = ACTIVITY_TO_LABEL[activity]

        # ── 모드 B: 어노테이션 기반 낙상 윈도우 ──────────────────────
        if is_fall and onset >= 0:
            raw = load_safesignal_csv(path, rx="both")
            res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
            raw_windows = _extract_annotated_fall(res.amplitude, onset, window_size)
            if raw_windows.shape[0] == 0:
                return path, None, "no_windows_annotated"
            inputs = windows_to_model_input(raw_windows)   # (n, 1, 28, 20)

        # ── 모드 A: stride 기반 ────────────────────────────────────────
        else:
            stride = fall_stride if is_fall else nonfail_stride
            result = preprocess_safesignal_file_full(
                path,
                window_size=window_size,
                stride=stride,
                drop_last=True,
                tail_window=False,
            )
            inputs = result.inputs
            if inputs.shape[0] == 0:
                return path, None, "no_windows"
            if is_fall:
                keep = [i for i in range(len(inputs)) if i * fall_stride <= FALL_MAX_START]
                if not keep:
                    keep = [0]
                inputs = inputs[keep]

        n_win = inputs.shape[0]
        records = [
            {
                "x":                 inputs[i],
                "y":                 label,
                "subject":           meta.subject,
                "env":               meta.environment,
                "activity":          activity,
                "trial":             meta.trial,
                "filename":          Path(path).name,
                "within_file_index": i,
            }
            for i in range(n_win)
        ]
        return path, records, None

    except Exception as exc:
        return path, None, repr(exc)


# ── 메인 ─────────────────────────────────────────────────────────────

def build_cache(
    src: Path,
    out: Path,
    fall_stride: int = 50,
    nonfail_stride: int = 100,
    n_workers: int | None = None,
    window_size: int = WINDOW_SIZE,
    annotations: dict[str, int] | None = None,
) -> None:
    paths = sorted(src.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"CSV 파일 없음: {src}")

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    mode = "어노테이션" if annotations else "stride"
    fall_annotated = sum(1 for p in paths if p.name in (annotations or {}))
    print(f"대상 파일: {len(paths)}개  모드: {mode}")
    if annotations:
        print(f"어노테이션 적용 낙상 파일: {fall_annotated}개")
    else:
        print(f"낙상 stride={fall_stride} (세션당 최대 {FALL_MAX_START // fall_stride + 1}윈도우)")
    print(f"비낙상 stride={nonfail_stride}  워커: {n_workers}")

    # onset: 어노테이션 있으면 값, 없으면 -1 (stride 모드)
    def _onset(p: Path) -> int:
        if annotations and p.name in annotations:
            return int(annotations[p.name])
        return -1

    args_list = [(str(p), fall_stride, nonfail_stride, window_size, _onset(p)) for p in paths]
    all_records: list[dict] = []
    skipped_err = 0
    skipped_cls = 0

    def _collect(path: str, records: list[dict] | None, err: str | None) -> None:
        nonlocal skipped_err, skipped_cls
        if err is not None:
            if err.startswith("skip:"):
                skipped_cls += 1
            else:
                tqdm.write(f"[오류] {Path(path).name}: {err}")
                skipped_err += 1
            return
        all_records.extend(records)

    if n_workers <= 1:
        for args in tqdm(args_list, desc="캐시 생성 (단일)"):
            _collect(*_worker(args))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_worker, a) for a in args_list]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"캐시 생성 ({n_workers}워커)"):
                _collect(*fut.result())

    if not all_records:
        raise RuntimeError("생성된 샘플이 없습니다.")

    print(f"\n총 샘플: {len(all_records)}  (클래스 스킵: {skipped_cls}, 오류 스킵: {skipped_err})")

    X        = np.stack([r["x"] for r in all_records]).astype(np.float32)
    y        = np.array([r["y"]       for r in all_records], dtype=np.int64)
    subject  = np.array([r["subject"] for r in all_records], dtype=np.int64)
    env      = np.array([r["env"]     for r in all_records], dtype=np.int64)
    activity = np.array([r["activity"] for r in all_records], dtype=object)
    trial    = np.array([r["trial"]   for r in all_records], dtype=np.int64)
    filename = np.array([r["filename"] for r in all_records], dtype=object)
    wfi      = np.array([r["within_file_index"] for r in all_records], dtype=np.int64)
    source   = np.array([SOURCE] * len(all_records), dtype=object)
    is_aug   = np.zeros(len(all_records), dtype=bool)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        X=X, y=y,
        subject=subject, source=source, env=env,
        filename=filename, activity=activity,
        trial=trial, within_file_index=wfi,
        is_augmented=is_aug,
        classes=np.array(CLASS_NAMES, dtype=object),
        class_policy=CLASS_POLICY,
        normalization=NORMALIZATION,
    )
    print(f"저장: {out}")

    print("\n[ 클래스별 샘플 수 ]")
    for i, name in enumerate(CLASS_NAMES):
        cnt = int((y == i).sum())
        bar = "#" * (cnt // 50)
        print(f"  {name:12s} ({i}): {cnt:5d}  {bar}")

    print("\n[ 낙상 세션별 윈도우 수 분포 ]")
    fall_mask = y == 0
    if fall_mask.any():
        fall_wfi = wfi[fall_mask]
        for w in sorted(np.unique(fall_wfi)):
            print(f"  within_file_index={w}: {int((fall_wfi == w).sum())}샘플")


if __name__ == "__main__":
    import json as _json
    parser = argparse.ArgumentParser(description="SafeSignal NPZ 캐시 생성")
    parser.add_argument("--src", type=Path, default=Path("dummy_src"))
    parser.add_argument("--out", type=Path,
                        default=Path("model/finetune/cache/ss_stride50_p6.npz"))
    parser.add_argument("--fall-stride", type=int, default=50)
    parser.add_argument("--nonfail-stride", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--annotations", type=Path, default=None,
                        help="fall_onsets.json 경로 (지정 시 어노테이션 모드)")
    args = parser.parse_args()

    ann = None
    if args.annotations and args.annotations.exists():
        ann = _json.loads(args.annotations.read_text(encoding="utf-8"))
        print(f"어노테이션 로드: {len(ann)}개")

    build_cache(
        src=args.src,
        out=args.out,
        fall_stride=args.fall_stride,
        nonfail_stride=args.nonfail_stride,
        n_workers=args.workers,
        annotations=ann,
    )
