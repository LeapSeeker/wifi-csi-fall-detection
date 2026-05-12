"""SDP 출력 shape 및 파라미터 조합 검증.

실행:
    python debug/preprocessing/check_sdp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model.preprocessing.sdp import stacked_doppler_profile, SUB_W, SUB_STRIDE, W_T
from model.preprocessing.acf import N_LAGS


def check_basic_shape() -> None:
    print("\n[1] 기본 shape 검증 (300, 90) → (28, 20)")
    rng = np.random.default_rng(0)
    mat = rng.uniform(-1, 1, (300, 90)).astype(np.float32)
    out = stacked_doppler_profile(mat)
    ok = out.shape == (W_T, N_LAGS) and out.dtype == np.float32
    print(f"  입력 {mat.shape} → 출력 {out.shape} dtype={out.dtype}  {'PASS' if ok else 'FAIL'}")


def check_n_sc_robustness() -> None:
    print("\n[2] n_sc 내성 검증 (다양한 서브캐리어 수)")
    rng = np.random.default_rng(0)
    for n_sc in [90, 104, 52]:
        mat = rng.uniform(-1, 1, (300, n_sc)).astype(np.float32)
        out = stacked_doppler_profile(mat)
        ok = out.shape == (W_T, N_LAGS)
        print(f"  n_sc={n_sc:>4}: 출력 shape={out.shape}  {'PASS' if ok else 'FAIL'}")


def check_wt_formula() -> None:
    print("\n[3] W_T 공식 검증: W_T = (n_t - sub_w) // stride + 1")
    rng = np.random.default_rng(0)
    cases = [
        (300, 30, 10),
        (300, 20, 5),
        (200, 30, 10),
        (400, 40, 20),
    ]
    print(f"  {'n_t':>5}  {'sub_w':>6}  {'stride':>7}  {'기대 W_T':>9}  {'실제 W_T':>9}  결과")
    for n_t, sub_w, stride in cases:
        mat = rng.uniform(-1, 1, (n_t, 52)).astype(np.float32)
        out = stacked_doppler_profile(mat, sub_w=sub_w, stride=stride)
        expected = (n_t - sub_w) // stride + 1
        ok = out.shape[0] == expected
        print(f"  {n_t:>5}  {sub_w:>6}  {stride:>7}  {expected:>9}  {out.shape[0]:>9}  {'PASS' if ok else 'FAIL'}")


def check_aggregate_comparison() -> None:
    print("\n[4] aggregate 비교 (mean/sum/max)")
    rng = np.random.default_rng(0)
    mat = rng.uniform(-1, 1, (300, 52)).astype(np.float32)
    print(f"  {'방식':>6}  {'min':>8}  {'max':>8}  {'mean':>8}  {'std':>8}")
    for mode in ["mean", "sum", "max"]:
        out = stacked_doppler_profile(mat, aggregate=mode)
        print(f"  {mode:>6}  {out.min():>8.4f}  {out.max():>8.4f}  "
              f"{out.mean():>8.4f}  {out.std():>8.4f}")
    print("  → mean 사용 이유: 서브캐리어 간 스케일 차이를 정규화하여 안정적인 학습 가능")


def check_exceptions() -> None:
    print("\n[5] 예외 처리 검증")
    rng = np.random.default_rng(0)

    # n_t < sub_w
    try:
        mat = rng.uniform(-1, 1, (10, 52)).astype(np.float32)
        stacked_doppler_profile(mat, sub_w=30)
        print("  n_t < sub_w: FAIL (예외 미발생)")
    except ValueError as e:
        print(f"  n_t < sub_w: PASS ({e})")

    # n_lags > sub_w
    try:
        mat = rng.uniform(-1, 1, (300, 52)).astype(np.float32)
        stacked_doppler_profile(mat, sub_w=15, n_lags=20)
        print("  n_lags > sub_w: FAIL (예외 미발생)")
    except ValueError as e:
        print(f"  n_lags > sub_w: PASS ({e})")

    # 3D 입력
    try:
        mat = rng.uniform(-1, 1, (300, 52, 2)).astype(np.float32)
        stacked_doppler_profile(mat)
        print("  3D 입력: FAIL (예외 미발생)")
    except ValueError as e:
        print(f"  3D 입력: PASS ({e})")


def main() -> None:
    print("=== SDP 검증 ===")
    check_basic_shape()
    check_n_sc_robustness()
    check_wt_formula()
    check_aggregate_comparison()
    check_exceptions()
    print("\nDONE")


if __name__ == "__main__":
    main()
