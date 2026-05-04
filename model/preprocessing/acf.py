"""자기상관(ACF) 계산.

서브윈도우 시계열의 lag-k 자기상관을 lag=0..N_Δ-1 까지 산출.
N_Δ = 20, ΔT = 0.01s (= 100Hz 샘플 간격).
정규화: rho_k = sum (x_t - mu)(x_{t+k} - mu) / sum (x_t - mu)^2  → rho_0 = 1.
"""
from __future__ import annotations

import numpy as np

N_LAGS = 20         # N_Δ
DELTA_T = 0.01      # s


def autocorrelation_1d(x: np.ndarray, n_lags: int = N_LAGS) -> np.ndarray:
    """1D 시계열의 ACF.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    n_lags : int
        반환 lag 개수 (lag=0..n_lags-1). n_lags <= len(x) 필요.

    Returns
    -------
    np.ndarray, float32, shape (n_lags,)
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n_lags > n:
        raise ValueError(f"n_lags({n_lags}) > len(x)({n})")

    xc = x - x.mean()
    denom = np.dot(xc, xc)
    out = np.empty(n_lags, dtype=np.float32)
    if denom == 0:
        out.fill(0.0)
        out[0] = 1.0
        return out

    for k in range(n_lags):
        if k == 0:
            out[k] = 1.0
        else:
            out[k] = np.dot(xc[: n - k], xc[k:]) / denom
    return out


def autocorrelation_matrix(matrix: np.ndarray, n_lags: int = N_LAGS) -> np.ndarray:
    """서브캐리어별 ACF.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_t, n_sc)
    n_lags : int

    Returns
    -------
    np.ndarray, float32, shape (n_sc, n_lags)
    """
    if matrix.ndim != 2:
        raise ValueError(f"2D 입력 필요. got shape={matrix.shape}")
    n_t, n_sc = matrix.shape
    out = np.empty((n_sc, n_lags), dtype=np.float32)
    for j in range(n_sc):
        out[j] = autocorrelation_1d(matrix[:, j], n_lags=n_lags)
    return out
