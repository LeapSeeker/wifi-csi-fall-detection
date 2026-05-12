"""RPCA 파라미터(λ, max_iter) 조정 및 수렴 충분 여부 확인.

실행:
    python debug/preprocessing/check_rpca.py                      # 합성 데이터
    python debug/preprocessing/check_rpca.py --csv path/to/file.csv  # 실제 CSV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model.preprocessing.rpca import rpca_sparse, DEFAULT_MAX_ITER
from model.preprocessing.r_pca import RobustPCA


def _load_matrix(csv_path: str | None) -> np.ndarray:
    if csv_path:
        import pandas as pd
        df = pd.read_csv(csv_path)
        amp_cols = [c for c in df.columns if c.startswith("amp_rx1_")]
        return df[amp_cols].values.astype(np.float32)
    # 합성 데이터: (300, 90)
    rng = np.random.default_rng(42)
    base = rng.uniform(0.1, 1.0, size=(300, 90)).astype(np.float32)
    sparse_gt = np.zeros_like(base)
    sparse_gt[100:120, 20:40] = rng.uniform(0.5, 1.5, (20, 20))
    return base + sparse_gt


def check_lambda_sensitivity(matrix: np.ndarray) -> None:
    print("\n[1] λ 민감도 검증")
    print(f"{'λ 배율':>8}  {'λ 값':>10}  {'sparse 에너지':>14}  {'비영 비율':>10}")
    n_t, n_s = matrix.shape
    base_lmbda = 1.0 / np.sqrt(max(n_t, n_s))
    for factor in [0.5, 1.0, 2.0]:
        lmbda = base_lmbda * factor
        D = matrix.astype(np.float64)
        rpca = RobustPCA(D, lmbda=lmbda)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            _, S = rpca.fit(max_iter=50, iter_print=9999)
        energy = float(np.linalg.norm(S, 'fro'))
        nonzero = float((np.abs(S) > 1e-4).mean())
        print(f"  {factor:>6.1f}x  {lmbda:>10.6f}  {energy:>14.4f}  {nonzero:>10.4f}")


def check_convergence(matrix: np.ndarray) -> None:
    print("\n[2] max_iter 수렴 검증")
    print(f"{'max_iter':>10}  {'재구성 오차':>14}")
    D_norm = float(np.linalg.norm(matrix, 'fro'))
    for iters in [10, 50, 100, 150, 200]:
        S = rpca_sparse(matrix, max_iter=iters)
        L = matrix - S
        err = float(np.linalg.norm(matrix - L - S, 'fro')) / max(D_norm, 1e-8)
        print(f"  {iters:>8}  {err:>14.8f}")


def check_shape_robustness() -> None:
    print("\n[3] 입력 shape 내성 검증")
    rng = np.random.default_rng(0)
    cases = [
        ("Alsaify (300, 90)", rng.uniform(0, 1, (300, 90)).astype(np.float32)),
        ("ESP32 concat (300, 104)", rng.uniform(0, 1, (300, 104)).astype(np.float32)),
        ("단일 Rx (300, 52)", rng.uniform(0, 1, (300, 52)).astype(np.float32)),
    ]
    for name, mat in cases:
        try:
            S = rpca_sparse(mat, max_iter=30)
            assert S.shape == mat.shape
            assert S.dtype == np.float32
            print(f"  {name}: PASS  shape={S.shape} dtype={S.dtype}")
        except Exception as e:
            print(f"  {name}: FAIL  {e}")


def check_sparse_quality(matrix: np.ndarray) -> None:
    print("\n[4] sparse 성분 품질")
    S = rpca_sparse(matrix, max_iter=DEFAULT_MAX_ITER)
    nonzero_ratio = float((np.abs(S) > 1e-4).mean())
    print(f"  shape     : {S.shape}")
    print(f"  dtype     : {S.dtype}")
    print(f"  min/max   : {S.min():.4f} / {S.max():.4f}")
    print(f"  std       : {S.std():.4f}")
    print(f"  비영 비율 : {nonzero_ratio:.4f}  (|S|>1e-4)")
    has_nan = bool(np.isnan(S).any())
    has_inf = bool(np.isinf(S).any())
    print(f"  NaN/Inf   : {has_nan} / {has_inf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RPCA 파라미터 검증")
    parser.add_argument("--csv", type=str, default=None, help="실제 CSV 경로")
    args = parser.parse_args()

    matrix = _load_matrix(args.csv)
    print(f"입력 행렬: shape={matrix.shape} dtype={matrix.dtype} "
          f"range=[{matrix.min():.3f}, {matrix.max():.3f}]")

    check_lambda_sensitivity(matrix)
    check_convergence(matrix)
    check_shape_robustness()
    check_sparse_quality(matrix)
    print("\nDONE")


if __name__ == "__main__":
    main()
