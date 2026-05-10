"""SDP (Stacked Doppler Profile) 집계.

서브윈도우 W=30, stride=10 으로 시간 축을 분할 → 각 서브윈도우마다 ACF (n_sc, N_Δ)
를 계산 → 서브캐리어 축으로 평균 → (N_Δ,) 1D 프로파일 → W_T 개를 stack.

300패킷 입력 기준:
  W_T = (300 - 30) / 10 + 1 = 28
  최종 출력: (28, 20) — 모델 (N, 1, 28, 20) 입력의 (28, 20) 부분.

서브캐리어 수가 데이터셋마다 달라도(Alsaify 90 vs ESP32 52) 이 단계에서
mean 집계로 흡수되므로 다운스트림(CNN+GRU+Attention) 입력 차원이 일정.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .acf import N_LAGS, autocorrelation_matrix

SUB_W = 30
SUB_STRIDE = 10
W_T = 28  # (300 - 30) / 10 + 1

Aggregation = Literal["mean", "sum", "max"]


def _aggregate(acf: np.ndarray, mode: Aggregation) -> np.ndarray:
    if mode == "mean":
        return acf.mean(axis=0)
    if mode == "sum":
        return acf.sum(axis=0)
    if mode == "max":
        return acf.max(axis=0)
    raise ValueError(f"unknown aggregate mode: {mode}")


def stacked_doppler_profile(
    matrix: np.ndarray,
    sub_w: int = SUB_W,
    stride: int = SUB_STRIDE,
    n_lags: int = N_LAGS,
    aggregate: Aggregation = "mean",
) -> np.ndarray:
    """서브윈도우 ACF 집계.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_t, n_sc)
        보통 RPCA sparse 성분 (300, 90) 또는 (300, 52).
    sub_w : int
    stride : int
    n_lags : int
    aggregate : "mean" | "sum" | "max"
        서브캐리어 축 집계 방식.

    Returns
    -------
    np.ndarray, float32, shape (W_T, n_lags)
        n_t=300, sub_w=30, stride=10 → (28, 20).
    """
    if matrix.ndim != 2:
        raise ValueError(f"2D 입력 필요. got shape={matrix.shape}")
    n_t, n_sc = matrix.shape
    if n_t < sub_w:
        raise ValueError(f"n_t({n_t}) < sub_w({sub_w})")
    if n_lags > sub_w:
        raise ValueError(f"n_lags({n_lags}) > sub_w({sub_w})")

    n_sub = (n_t - sub_w) // stride + 1
    out = np.empty((n_sub, n_lags), dtype=np.float32)
    for i in range(n_sub):
        s = i * stride
        sub = matrix[s : s + sub_w]                       # (sub_w, n_sc)
        acf = autocorrelation_matrix(sub, n_lags=n_lags)  # (n_sc, n_lags)
        out[i] = _aggregate(acf, aggregate)
    return out
