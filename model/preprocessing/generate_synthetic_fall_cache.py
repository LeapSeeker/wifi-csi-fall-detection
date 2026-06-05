"""도메인 특성 기반 순수 합성 fall 패턴 생성.

원리
----
낙상의 SDP 특성은 정규화 후에도 시간에 따른 subcarrier 패턴 변화로 구별된다:
  - 낙상 전: 사람의 주요 활동 패턴 (정규화 후 특정 subcarrier에 집중)
  - 낙상 순간: 패턴 급변 (전환점)
  - 낙상 후: 바닥 눕기의 새로운 안정 패턴

이 스크립트는 training fall 샘플에서 각 phase 통계를 추출하고,
fall onset이 임의 위치 k에 있는 합성 창을 생성한다.
test 세션은 split_seed42.json으로 완전히 제외한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

T      = 28   # SDP 시간 스텝 수
N_SC   = 20   # subcarrier 수
FALL_CLS = 0


def _extract_phase_stats(
    X: np.ndarray,     # (N, 1, 28, 20) training fall samples
    onset_step: int = 9,  # wfi=0 기준 onset 위치
) -> dict:
    """각 phase의 평균·공분산 통계 추출."""
    x = X[:, 0, :, :]  # (N, 28, 20)

    pre  = x[:, :onset_step,       :]   # (N, onset,      20)
    spk  = x[:, onset_step:onset_step+3, :]  # (N, 3, 20) — 전환 구간
    post = x[:, onset_step+3:,      :]  # (N, rest, 20)

    def phase_stat(arr):  # arr: (N, t, F)
        flat = arr.reshape(-1, N_SC)     # (N*t, 20)
        return flat.mean(axis=0), flat.std(axis=0)

    pre_m,  pre_s  = phase_stat(pre)
    spk_m,  spk_s  = phase_stat(spk)
    post_m, post_s = phase_stat(post)

    return {
        "pre_mean":  pre_m,   "pre_std":  pre_s,
        "spk_mean":  spk_m,   "spk_std":  spk_s,
        "post_mean": post_m,  "post_std": post_s,
    }


def _synthesize_fall(
    stats: dict,
    onset: int,      # fall이 시작되는 SDP 스텝 (0~T-5)
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """onset 위치에서 낙상이 발생하는 (1, 28, 20) SDP 샘플 합성."""
    x = np.zeros((1, T, N_SC), dtype=np.float32)

    pre_len  = onset
    spk_len  = min(3, T - onset)
    post_len = T - onset - spk_len

    if pre_len > 0:
        x[0, :pre_len, :]              = (stats["pre_mean"]
                                         + rng.normal(0, stats["pre_std"] * noise_scale, (pre_len, N_SC)))
    if spk_len > 0:
        x[0, onset:onset+spk_len, :]  = (stats["spk_mean"]
                                         + rng.normal(0, stats["spk_std"] * noise_scale, (spk_len, N_SC)))
    if post_len > 0:
        x[0, onset+spk_len:, :]       = (stats["post_mean"]
                                         + rng.normal(0, stats["post_std"] * noise_scale, (post_len, N_SC)))

    return x.astype(np.float32)


def build_synthetic_cache(
    cache_path: Path,
    split_path: Path,
    out_path: Path,
    n_per_position: int,
    onset_range: tuple[int, int],   # (min, max) onset 범위
    noise_scale: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    # ── split 로드 ──────────────────────────────────────────────────────
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_filenames: set[str] = set(split["train"])

    # ── 캐시 로드 ────────────────────────────────────────────────────────
    d = np.load(cache_path, allow_pickle=True)
    X         = d["X"]
    y         = d["y"]
    filenames = d["filename"]
    wfi       = d["within_file_index"]

    # ── train 낙상 샘플 (wfi=0) 만 phase stats 추출에 사용 ─────────────
    train_mask = np.array([fn in train_filenames for fn in filenames])
    fall_wfi0_mask = train_mask & (y == FALL_CLS) & (wfi == 0)
    X_fall = X[fall_wfi0_mask]
    print(f"Phase stats 추출용 fall wfi=0 샘플: {len(X_fall)}")

    stats = _extract_phase_stats(X_fall, onset_step=9)
    print(f"  pre_mean range: [{stats['pre_mean'].min():.3f}, {stats['pre_mean'].max():.3f}]")
    print(f"  spk_mean range: [{stats['spk_mean'].min():.3f}, {stats['spk_mean'].max():.3f}]")
    print(f"  post_mean range: [{stats['post_mean'].min():.3f}, {stats['post_mean'].max():.3f}]")

    # ── 합성 ────────────────────────────────────────────────────────────
    onset_positions = list(range(onset_range[0], onset_range[1] + 1))
    total = len(onset_positions) * n_per_position
    print(f"\nonset 범위: {onset_range}  위치수: {len(onset_positions)}  "
          f"위치당 샘플: {n_per_position}  예상 총계: {total}")

    X_out    = np.zeros((total, 1, T, N_SC), dtype=np.float32)
    onset_arr= np.zeros(total, dtype=np.int64)
    i = 0
    for onset in onset_positions:
        for _ in range(n_per_position):
            X_out[i]     = _synthesize_fall(stats, onset, rng, noise_scale)
            onset_arr[i] = onset
            i += 1

    y_out       = np.zeros(total, dtype=np.int64)
    fn_out      = np.array([f"SYN_onset{o:02d}_n{n:04d}"
                            for o in onset_positions
                            for n in range(n_per_position)], dtype=object)
    is_aug      = np.ones(total, dtype=bool)
    source      = np.array(["synthetic_domain"] * total, dtype=object)
    subj_out    = np.zeros(total, dtype=np.int64)
    env_out     = np.zeros(total, dtype=np.int64)
    act_out     = np.array(["SYNTHETIC_FALL"] * total, dtype=object)
    trial_out   = np.zeros(total, dtype=np.int64)
    wfi_out     = onset_arr  # onset position을 wfi로 기록

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X=X_out, y=y_out,
        filename=fn_out, origin_filename=fn_out,
        onset_step=onset_arr,
        subject=subj_out, source=source, env=env_out,
        activity=act_out, trial=trial_out, within_file_index=wfi_out,
        is_augmented=is_aug,
        classes=d["classes"],
        class_policy="pretrained6",
        normalization="global_self",
        build_params=np.array([
            f"onset_range={onset_range}",
            f"n_per_position={n_per_position}",
            f"noise_scale={noise_scale}",
            f"seed={seed}",
        ], dtype=object),
    )
    print(f"\n저장: {out_path}  ({total} 샘플)")
    print(f"onset 분포: {onset_range[0]}~{onset_range[1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="도메인 특성 기반 합성 fall 캐시 생성")
    ap.add_argument("--cache",  type=Path,
                    default=Path("model/finetune/cache/ss_peak160_p6.npz"))
    ap.add_argument("--split",  type=Path,
                    default=Path("model/finetune/cache/split_seed42.json"))
    ap.add_argument("--out",    type=Path,
                    default=Path("model/finetune/cache/aug_fall_synthetic.npz"))
    ap.add_argument("--n-per-position", type=int, default=50,
                    help="onset 위치당 합성 샘플 수 (기본 50)")
    ap.add_argument("--onset-min",  type=int, default=0)
    ap.add_argument("--onset-max",  type=int, default=23)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    build_synthetic_cache(
        cache_path=args.cache,
        split_path=args.split,
        out_path=args.out,
        n_per_position=args.n_per_position,
        onset_range=(args.onset_min, args.onset_max),
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
