"""낙상 세션 raw 진폭 단계 증강 → 파이프라인 재계산 → NPZ 캐시.

재설계 원칙
-----------
1. raw CSI 진폭 (300, 104) 에 증강 적용 → RPCA→ACF→SDP→z-score 재계산
2. --split JSON 으로 train 세션만 필터링 (test/val 오염 원천 차단)
3. is_augmented=True 로 정직하게 표기 (게이트 우회 금지)
4. origin_filename / augment_type 메타데이터 기록

사용:
    # 1단계: split 먼저 고정 (model/finetune/export_split.py)
    python -m model.finetune.export_split \\
        --cache model/finetune/cache/safesignal_e1234_pretrained6.npz \\
        --out model/finetune/cache/split_seed42.json

    # 2단계: train 세션만 증강
    python -m model.preprocessing.generate_dummy_fall_cache \\
        --src dummy_src \\
        --split model/finetune/cache/split_seed42.json \\
        --out model/finetune/cache/aug_fall_raw.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.preprocessing.loader import load_safesignal_csv, parse_safesignal_filename
from model.preprocessing.pipeline import windows_to_model_input
from model.preprocessing.resample import resample_to_100hz

FALL_ACTIVITIES = {
    "FALL_SIT_B", "FALL_SIT_F",
    "FALL_STD_B", "FALL_STD_F",
    "FALL_WALK_B", "FALL_WALK_F",
}

FALL_LABEL = 0
WINDOW_SIZE = 300
# 낙상 피크 구간: 1.0~2.5s 사이 3개 오프셋으로 윈도우 추출
FALL_SHIFTS = (-100, -50, 0)   # onset 기준 시작 오프셋 (패킷)
FALL_ONSET_DEFAULT = 160       # 수집 프로토콜 기준 1.6s @ 100Hz

CLASS_NAMES = ("fall", "walking", "sit_stand", "lying", "standing", "picking")
SOURCE = "safesignal"


# ── raw 진폭 (T, F) 전용 증강 ──────────────────────────────────────────────

def _raw_jitter(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return (x + rng.normal(0.0, sigma, size=x.shape)).astype(np.float32)


def _raw_scale(x: np.ndarray, scale: float) -> np.ndarray:
    return (x * scale).astype(np.float32)


def _raw_time_warp(
    x: np.ndarray, sigma: float, knot: int, rng: np.random.Generator
) -> np.ndarray:
    """(T, F) 배열의 시간 축(axis=0) 왜곡."""
    n_t = x.shape[0]
    knot_xs = np.linspace(0, n_t - 1, knot + 2)
    knot_ys = rng.normal(loc=1.0, scale=sigma, size=knot + 2)
    cs = CubicSpline(knot_xs, knot_ys)
    distortion = np.maximum(cs(np.arange(n_t)), 1e-3)
    tt_cum = np.cumsum(distortion)
    tt_cum = (tt_cum - tt_cum[0]) / (tt_cum[-1] - tt_cum[0]) * (n_t - 1)
    time_steps = np.arange(n_t)
    out = np.empty_like(x)
    for f in range(x.shape[1]):
        out[:, f] = np.interp(time_steps, tt_cum, x[:, f])
    return out.astype(np.float32)


def _raw_noise_scale(
    x: np.ndarray, sigma: float, scale: float, rng: np.random.Generator
) -> np.ndarray:
    return _raw_scale(_raw_jitter(x, sigma, rng), scale)


def _apply_augmentations(
    window: np.ndarray,
    rng: np.random.Generator,
    jitter_sigma: float,
    scale_range: tuple[float, float],
    warp_sigma: float,
    warp_knot: int,
) -> list[tuple[str, np.ndarray]]:
    """(300, 104) raw 진폭에 4기법 적용 → (augment_type, augmented_window) 리스트."""
    scale = float(rng.uniform(*scale_range))
    return [
        ("jitter",     _raw_jitter(window, jitter_sigma, rng)),
        ("scale",      _raw_scale(window, scale)),
        ("timewarp",   _raw_time_warp(window, warp_sigma, warp_knot, rng)),
        ("noise_scale", _raw_noise_scale(window, jitter_sigma, scale, rng)),
    ]


# ── 윈도우 추출 ────────────────────────────────────────────────────────────

def _extract_windows(amp: np.ndarray, onset: int) -> list[np.ndarray]:
    """onset 기준 FALL_SHIFTS 오프셋으로 윈도우 추출."""
    n = len(amp)
    windows = []
    for shift in FALL_SHIFTS:
        start = onset + shift
        end = start + WINDOW_SIZE
        if start < 0:
            start, end = 0, WINDOW_SIZE
        if end > n:
            start, end = n - WINDOW_SIZE, n
        if start < 0:
            continue
        windows.append(amp[start:end].astype(np.float32))
    return windows


# ── 메인 빌드 ──────────────────────────────────────────────────────────────

def build_cache(
    src: Path,
    out: Path,
    train_filenames: set[str] | None,
    jitter_sigma: float,
    scale_range: tuple[float, float],
    warp_sigma: float,
    warp_knot: int,
    seed: int,
) -> None:
    fall_paths = [
        p for p in sorted(src.rglob("*.csv"))
        if parse_safesignal_filename(p).activity.upper() in FALL_ACTIVITIES
    ]

    if train_filenames is not None:
        before = len(fall_paths)
        fall_paths = [p for p in fall_paths if p.name in train_filenames]
        print(f"split 필터: {before} → {len(fall_paths)} (train 세션만)")

    if not fall_paths:
        raise FileNotFoundError(f"낙상 CSV 없음 (필터 후): {src}")

    print(f"낙상 세션: {len(fall_paths)}  "
          f"파라미터: jitter_sigma={jitter_sigma} scale={scale_range} "
          f"warp_sigma={warp_sigma} warp_knot={warp_knot}")

    rng = np.random.default_rng(seed)
    records: list[dict] = []
    skipped = 0

    for p in fall_paths:
        meta = parse_safesignal_filename(p)
        try:
            raw = load_safesignal_csv(p, rx="both")
            res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
            amp = res.amplitude
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
            skipped += 1
            continue

        raw_windows = _extract_windows(amp, FALL_ONSET_DEFAULT)
        if not raw_windows:
            skipped += 1
            continue

        for win_idx, raw_win in enumerate(raw_windows):
            augs = _apply_augmentations(
                raw_win, rng, jitter_sigma, scale_range, warp_sigma, warp_knot
            )
            for aug_type, aug_win in augs:
                try:
                    # RPCA → ACF → SDP → z-score (파이프라인 재계산)
                    inp = windows_to_model_input(aug_win[None])  # (1, 1, 28, 20)
                    records.append({
                        "x":               inp[0],       # (1, 28, 20)
                        "y":               FALL_LABEL,
                        "subject":         meta.subject,
                        "env":             meta.environment,
                        "filename":        f"AUG_{aug_type}_{p.name}",
                        "origin_filename": p.name,
                        "augment_type":    aug_type,
                        "activity":        meta.activity.upper(),
                        "trial":           meta.trial,
                        "within_file_index": win_idx,
                    })
                except Exception:
                    continue

    if not records:
        raise RuntimeError("생성된 샘플 없음")

    print(f"생성: {len(records)}  스킵: {skipped}")

    X               = np.stack([r["x"]               for r in records]).astype(np.float32)
    y               = np.array([r["y"]               for r in records], dtype=np.int64)
    subject         = np.array([r["subject"]         for r in records], dtype=np.int64)
    env             = np.array([r["env"]             for r in records], dtype=np.int64)
    filename        = np.array([r["filename"]        for r in records], dtype=object)
    origin_filename = np.array([r["origin_filename"] for r in records], dtype=object)
    augment_type    = np.array([r["augment_type"]    for r in records], dtype=object)
    activity        = np.array([r["activity"]        for r in records], dtype=object)
    trial           = np.array([r["trial"]           for r in records], dtype=np.int64)
    wfi             = np.array([r["within_file_index"] for r in records], dtype=np.int64)
    source          = np.array([SOURCE] * len(records), dtype=object)
    is_aug          = np.ones(len(records), dtype=bool)   # 항상 True

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        X=X, y=y,
        subject=subject, source=source, env=env,
        filename=filename, origin_filename=origin_filename,
        augment_type=augment_type, activity=activity,
        trial=trial, within_file_index=wfi,
        is_augmented=is_aug,
        classes=np.array(CLASS_NAMES, dtype=object),
        class_policy="pretrained6",
        normalization="global_self",
        build_params=np.array([
            f"jitter_sigma={jitter_sigma}",
            f"scale_range={scale_range}",
            f"warp_sigma={warp_sigma}",
            f"warp_knot={warp_knot}",
            f"seed={seed}",
        ], dtype=object),
    )
    print(f"저장: {out}")

    # 검증용 통계 출력
    print("\n[ augment_type별 샘플 수 ]")
    for t in sorted(set(r["augment_type"] for r in records)):
        cnt = sum(1 for r in records if r["augment_type"] == t)
        print(f"  {t:12s}: {cnt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="낙상 raw 진폭 증강 캐시 생성")
    ap.add_argument("--src",  type=Path, default=Path("dummy_src"))
    ap.add_argument("--out",  type=Path,
                    default=Path("model/finetune/cache/aug_fall_raw.npz"))
    ap.add_argument("--split", type=Path, default=None,
                    help="export_split.py 출력 JSON. 미지정 시 split 필터 없음(비권장)")
    ap.add_argument("--jitter-sigma", type=float, default=0.05)
    ap.add_argument("--scale-min",   type=float, default=0.8)
    ap.add_argument("--scale-max",   type=float, default=1.2)
    ap.add_argument("--warp-sigma",  type=float, default=0.05)
    ap.add_argument("--warp-knot",   type=int,   default=4)
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    train_filenames = None
    if args.split is not None:
        split = json.loads(args.split.read_text(encoding="utf-8"))
        train_filenames = set(split["train"])
        print(f"split 로드: train={len(train_filenames)} val={len(split['val'])} "
              f"test={len(split['test'])} (seed={split.get('seed')})")
    else:
        print("경고: --split 미지정 → train 세션 필터 없음. test 오염 위험.")

    build_cache(
        src=args.src,
        out=args.out,
        train_filenames=train_filenames,
        jitter_sigma=args.jitter_sigma,
        scale_range=(args.scale_min, args.scale_max),
        warp_sigma=args.warp_sigma,
        warp_knot=args.warp_knot,
        seed=args.seed,
    )
