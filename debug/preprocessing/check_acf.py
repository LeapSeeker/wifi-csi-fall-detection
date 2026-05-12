"""ACF 계산 수치 정확성 및 파라미터 영향 확인.

실행:
    python debug/preprocessing/check_acf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model.preprocessing.acf import autocorrelation_1d, autocorrelation_matrix, N_LAGS


def check_boundary_conditions() -> None:
    print("\n[1] 경계 조건 검증 (rho_0=1, |rho_k|<=1)")
    rng = np.random.default_rng(42)
    all_pass = True
    for i in range(10):
        x = rng.uniform(-1, 1, size=100).astype(np.float32)
        acf = autocorrelation_1d(x, n_lags=N_LAGS)
        if not np.isclose(acf[0], 1.0, atol=1e-5):
            print(f"  [FAIL] 신호 {i}: rho_0={acf[0]:.6f} != 1")
            all_pass = False
        if not (np.abs(acf) <= 1.0 + 1e-5).all():
            print(f"  [FAIL] 신호 {i}: |rho_k| > 1 발생")
            all_pass = False
    if all_pass:
        print("  10개 랜덤 신호 모두 PASS")


def check_known_signal() -> None:
    print("\n[2] 알려진 신호 검증 (2Hz 사인파)")
    t = np.linspace(0, 1, 100, endpoint=False)
    x = np.sin(2 * np.pi * 2 * t).astype(np.float32)
    acf = autocorrelation_1d(x, n_lags=20)
    print(f"  rho_0={acf[0]:.4f}  rho_5={acf[5]:.4f}  rho_10={acf[10]:.4f}  rho_19={acf[19]:.4f}")
    print(f"  rho_0=1: {'PASS' if np.isclose(acf[0], 1.0, atol=1e-5) else 'FAIL'}")


def check_zero_signal() -> None:
    print("\n[3] 영 신호 (상수) 입력 검증")
    x = np.ones(100, dtype=np.float32) * 3.0
    acf = autocorrelation_1d(x, n_lags=N_LAGS)
    print(f"  rho_0={acf[0]:.4f} (기대 1.0)")
    print(f"  rho_k(k>0) 최대값={np.abs(acf[1:]).max():.6f} (기대 0.0)")
    print(f"  {'PASS' if np.isclose(acf[0], 1.0) and np.abs(acf[1:]).max() < 1e-5 else 'FAIL'}")


def check_n_lags_shape() -> None:
    print("\n[4] n_lags shape 검증")
    x = np.random.default_rng(0).uniform(-1, 1, 100).astype(np.float32)
    print(f"  {'n_lags':>8}  {'shape':>12}  {'dtype':>10}")
    for n in [5, 10, 20, 30]:
        acf = autocorrelation_1d(x, n_lags=n)
        ok = acf.shape == (n,) and acf.dtype == np.float32
        print(f"  {n:>8}  {str(acf.shape):>12}  {str(acf.dtype):>10}  {'PASS' if ok else 'FAIL'}")


def check_matrix_version() -> None:
    print("\n[5] autocorrelation_matrix 검증")
    rng = np.random.default_rng(0)
    cases = [
        ("(30, 90)",  rng.uniform(-1, 1, (30, 90)).astype(np.float32)),
        ("(30, 104)", rng.uniform(-1, 1, (30, 104)).astype(np.float32)),
        ("(30, 52)",  rng.uniform(-1, 1, (30, 52)).astype(np.float32)),
    ]
    for name, mat in cases:
        result = autocorrelation_matrix(mat, n_lags=N_LAGS)
        mean_acf = result.mean(axis=0)
        ok = result.shape == (mat.shape[1], N_LAGS) and mean_acf.shape == (N_LAGS,)
        print(f"  입력 {name} → matrix shape={result.shape} "
              f"mean shape={mean_acf.shape}  {'PASS' if ok else 'FAIL'}")


def main() -> None:
    print("=== ACF 검증 ===")
    check_boundary_conditions()
    check_known_signal()
    check_zero_signal()
    check_n_lags_shape()
    check_matrix_version()
    print("\nDONE")


if __name__ == "__main__":
    main()
