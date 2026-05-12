"""4가지 증강 기법의 shape/dtype 보존 및 파라미터 민감도 확인.

실행:
    python debug/augmentation/check_augment_shape.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model.augment.augment import jittering, scaling, time_warping, noise_scale, augment_all


def check_shape_dtype() -> None:
    print("\n[1] shape/dtype 보존 검증")
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, (1, 28, 20)).astype(np.float32)
    for fn in [jittering, scaling, time_warping, noise_scale]:
        y = fn(x, rng=np.random.default_rng(0))
        has_nan = bool(np.isnan(y).any())
        has_inf = bool(np.isinf(y).any())
        ok = y.shape == (1, 28, 20) and y.dtype == np.float32 and not has_nan and not has_inf
        print(f"  {fn.__name__:<15}: shape={y.shape} dtype={y.dtype} "
              f"NaN={has_nan} Inf={has_inf}  {'PASS' if ok else 'FAIL'}")


def check_jittering_sigma() -> None:
    print("\n[2] jittering sigma 민감도")
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (1, 28, 20)).astype(np.float32)
    print(f"  {'sigma':>8}  {'mean|Δ|':>10}  {'std|Δ|':>10}  {'max|Δ|':>10}")
    for sigma in [0.01, 0.05, 0.10, 0.20]:
        y = jittering(x, sigma=sigma, rng=np.random.default_rng(1))
        diff = np.abs(y - x)
        print(f"  {sigma:>8.2f}  {diff.mean():>10.6f}  {diff.std():>10.6f}  {diff.max():>10.6f}")


def check_scaling_range() -> None:
    print("\n[3] scaling range 분포 (50회 샘플)")
    rng_base = np.random.default_rng(0)
    x = rng_base.uniform(-1, 1, (1, 28, 20)).astype(np.float32)
    nz = np.abs(x) > 1e-3
    print(f"  {'scale_range':>15}  {'min scale':>10}  {'max scale':>10}  {'mean scale':>11}")
    for lo, hi in [(0.9, 1.1), (0.8, 1.2), (0.6, 1.4)]:
        scales = []
        for i in range(50):
            y = scaling(x, scale_range=(lo, hi), rng=np.random.default_rng(i))
            ratio = float((y[nz] / x[nz]).mean())
            scales.append(ratio)
        scales = np.array(scales)
        print(f"  ({lo:.1f}, {hi:.1f}):       {scales.min():>10.4f}  {scales.max():>10.4f}  {scales.mean():>11.4f}")


def check_time_warping_sigma() -> None:
    print("\n[4] time_warping sigma 민감도")
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (1, 28, 20)).astype(np.float32)
    print(f"  {'sigma':>8}  {'mean|Δ|':>10}")
    for sigma in [0.1, 0.2, 0.4]:
        y = time_warping(x, sigma=sigma, rng=np.random.default_rng(1))
        diff = float(np.abs(y - x).mean())
        print(f"  {sigma:>8.1f}  {diff:>10.6f}")


def check_determinism() -> None:
    print("\n[5] 결정론 검증")
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (1, 28, 20)).astype(np.float32)

    a1 = jittering(x, rng=np.random.default_rng(123))
    a2 = jittering(x, rng=np.random.default_rng(123))
    same_seed = np.array_equal(a1, a2)

    b1 = jittering(x, rng=np.random.default_rng(1))
    b2 = jittering(x, rng=np.random.default_rng(2))
    diff_seed = not np.array_equal(b1, b2)

    print(f"  동일 seed → 동일 결과: {'PASS' if same_seed else 'FAIL'}")
    print(f"  다른 seed → 다른 결과: {'PASS' if diff_seed else 'FAIL'}")


def check_augment_all() -> None:
    print("\n[6] augment_all() 검증")
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (1, 28, 20)).astype(np.float32)
    augs = augment_all(x, rng=np.random.default_rng(42))
    print(f"  길이: {len(augs)} (기대 5)  {'PASS' if len(augs) == 5 else 'FAIL'}")
    print(f"  [0]은 원본 참조: {'PASS' if augs[0] is x else 'FAIL'}")
    for i in range(1, 5):
        diff = not np.array_equal(augs[i], x)
        ok = augs[i].shape == (1, 28, 20) and augs[i].dtype == np.float32 and diff
        print(f"  [{i}] shape/dtype/변형: {'PASS' if ok else 'FAIL'}")


def main() -> None:
    print("=== 증강 shape/dtype 검증 ===")
    check_shape_dtype()
    check_jittering_sigma()
    check_scaling_range()
    check_time_warping_sigma()
    check_determinism()
    check_augment_all()
    print("\nDONE")


if __name__ == "__main__":
    main()
