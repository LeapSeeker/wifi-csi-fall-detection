"""증강 샘플의 다양성과 클래스 보존성 정량 평가.

해석 기준:
  cosine sim > 0.99 → 다양성 부족
  cosine sim < 0.80 → 클래스 특성 파괴 위험
  목표: 0.85 ~ 0.98 (실제 데이터로 재확인 필요)

실행:
    python debug/augmentation/check_augment_quality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model.augment.augment import jittering, scaling, time_warping, noise_scale, augment_all


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-10:
        return 1.0
    return float(np.dot(a_flat, b_flat) / denom)


def check_distance_from_original() -> None:
    print("\n[1] 원본 대비 거리 (100회 평균)")
    print(f"  {'기법':>15}  {'Frobenius norm':>16}  {'cosine sim':>12}  판정")

    # cosine sim 해석 기준
    def judge(sim: float) -> str:
        if sim > 0.99:
            return "다양성 부족"
        if sim < 0.80:
            return "클래스 파괴 위험"
        return "적정 (0.85~0.98)"

    rng_base = np.random.default_rng(0)
    x = rng_base.uniform(-1, 1, (1, 28, 20)).astype(np.float32)

    fns = [
        ("jittering", jittering),
        ("scaling", scaling),
        ("time_warping", time_warping),
        ("noise_scale", noise_scale),
    ]
    for name, fn in fns:
        frob_list, cos_list = [], []
        for i in range(100):
            y = fn(x, rng=np.random.default_rng(i))
            frob_list.append(float(np.linalg.norm((y - x).flatten())))
            cos_list.append(cosine_sim(x, y))
        frob_mean = np.mean(frob_list)
        cos_mean = np.mean(cos_list)
        print(f"  {name:>15}  {frob_mean:>16.6f}  {cos_mean:>12.6f}  {judge(cos_mean)}")


def check_pairwise_diversity() -> None:
    print("\n[2] augment_all() pairwise Frobenius norm 분포")
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, (1, 28, 20)).astype(np.float32)
    augs = augment_all(x, rng=np.random.default_rng(0))

    names = ["원본", "jittering", "scaling", "time_warping", "noise_scale"]
    dists = []
    pairs = []
    for i in range(5):
        for j in range(i + 1, 5):
            d = float(np.linalg.norm((augs[i] - augs[j]).flatten()))
            dists.append(d)
            pairs.append(f"{names[i]}-{names[j]}")

    dists = np.array(dists)
    print(f"  min={dists.min():.4f}  max={dists.max():.4f}  "
          f"mean={dists.mean():.4f}  std={dists.std():.4f}")
    print(f"  가장 가까운 쌍: {pairs[int(np.argmin(dists))]} ({dists.min():.4f})")
    print(f"  가장 먼 쌍:     {pairs[int(np.argmax(dists))]} ({dists.max():.4f})")


def check_distribution_preservation() -> None:
    print("\n[3] 분포 보존 (증강 전후 통계 비교)")
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (1, 28, 20)).astype(np.float32)
    print(f"  {'기법':>15}  {'mean 변화':>12}  {'std 비율':>10}  {'range 비율':>12}")
    fns = [
        ("jittering", jittering),
        ("scaling", scaling),
        ("time_warping", time_warping),
        ("noise_scale", noise_scale),
    ]
    x_range = float(x.max() - x.min())
    for name, fn in fns:
        y = fn(x, rng=np.random.default_rng(1))
        mean_delta = float(abs(y.mean() - x.mean()))
        std_ratio = float(y.std() / max(x.std(), 1e-8))
        y_range = float(y.max() - y.min())
        range_ratio = y_range / max(x_range, 1e-8)
        print(f"  {name:>15}  {mean_delta:>12.6f}  {std_ratio:>10.4f}  {range_ratio:>12.4f}")


def check_5x_augmentation() -> None:
    print("\n[4] 5× 증강 후 전체 통계")
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (1, 28, 20)).astype(np.float32)
    augs = augment_all(x, rng=np.random.default_rng(42))
    all_arr = np.stack(augs, axis=0)  # (5, 1, 28, 20)
    print(f"  shape: {all_arr.shape}")
    print(f"  mean={all_arr.mean():.4f}  std={all_arr.std():.4f}  "
          f"min={all_arr.min():.4f}  max={all_arr.max():.4f}")
    has_nan = bool(np.isnan(all_arr).any())
    has_inf = bool(np.isinf(all_arr).any())
    print(f"  NaN={has_nan}  Inf={has_inf}  {'PASS' if not has_nan and not has_inf else 'FAIL'}")


def main() -> None:
    print("=== 증강 품질 검증 ===")
    check_distance_from_original()
    check_pairwise_diversity()
    check_distribution_preservation()
    check_5x_augmentation()
    print("\nDONE")


if __name__ == "__main__":
    main()
