"""RPCA (Robust PCA) 래퍼.

CSI 진폭 행렬 D = L + S 분해:
- L (low-rank): 정적 배경 (가구, 벽 반사 등)
- S (sparse) : 동적 변화 (사람 동작, 낙상 이벤트)

낙상/활동 인식에는 S 성분만 사용한다.
λ = 1/sqrt(max(N_T, N_S)) — model/CLAUDE.md 전처리 표 준수.
구현체는 model/preprocessing/r_pca.py (Candès 2011 ADMM).
"""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np

from .r_pca import RobustPCA

DEFAULT_MAX_ITER = 200


def rpca_sparse(
    matrix: np.ndarray,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """D = L + S 분해 후 S(sparse) 성분 반환.

    Parameters
    ----------
    matrix : np.ndarray, shape (N_T, N_S)
        진폭 행렬 (시간 x 서브캐리어).
    max_iter : int
        ADMM 최대 반복.
    tol : float | None
        수렴 임계. None이면 1e-7 * frobenius(D).
    verbose : bool
        False면 r_pca.fit() 내부 print 억제.

    Returns
    -------
    sparse : np.ndarray, float32, shape (N_T, N_S)
    """
    if matrix.ndim != 2:
        raise ValueError(f"2D 입력 필요. got shape={matrix.shape}")

    n_t, n_s = matrix.shape
    lmbda = 1.0 / np.sqrt(max(n_t, n_s))

    D = np.asarray(matrix, dtype=np.float64)
    rpca = RobustPCA(D, lmbda=lmbda)

    iter_print = 100 if verbose else max(max_iter + 10, 10_000)
    sink = io.StringIO() if not verbose else None
    cm = contextlib.redirect_stdout(sink) if sink is not None else contextlib.nullcontext()
    with cm:
        _, S = rpca.fit(tol=tol, max_iter=max_iter, iter_print=iter_print)

    return S.astype(np.float32, copy=False)
