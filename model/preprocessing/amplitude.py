"""복소수 CSI에서 진폭 추출."""
from __future__ import annotations

import numpy as np


def to_amplitude(csi_complex: np.ndarray) -> np.ndarray:
    """sqrt(I^2 + Q^2). np.abs와 동치이지만 의도를 명시.

    Parameters
    ----------
    csi_complex : np.ndarray, complex dtype
        임의 shape. 일반적으로 (n_packets, 90).

    Returns
    -------
    amplitude : np.ndarray, float32
        입력과 동일 shape.
    """
    if not np.iscomplexobj(csi_complex):
        raise TypeError(f"복소수 배열이 아님: dtype={csi_complex.dtype}")
    return np.abs(csi_complex).astype(np.float32)
