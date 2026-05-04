"""300 패킷(=3초 @100Hz) 슬라이딩 윈도우 분절."""
from __future__ import annotations

import numpy as np

WINDOW_SIZE = 300  # 패킷 (3초 @100Hz, SHARED.md 모델 입력 스펙)


def sliding_windows(
    amplitude: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    drop_last: bool = True,
) -> np.ndarray:
    """시간 축을 따라 윈도우 분절.

    Parameters
    ----------
    amplitude : np.ndarray, shape (n_packets, n_subcarriers)
    window_size : int
        한 윈도우 패킷 수. 기본 300.
    stride : int | None
        없으면 window_size (=비중첩). 100이면 200패킷 중첩.
    drop_last : bool
        마지막 잔여 구간을 버림. False면 0 패딩.

    Returns
    -------
    np.ndarray, shape (n_windows, window_size, n_subcarriers)
        n_packets < window_size 면 (0, window_size, n_subcarriers) 반환.
    """
    if amplitude.ndim != 2:
        raise ValueError(f"2D 입력 필요 (n_packets, n_sc). got shape={amplitude.shape}")
    if stride is None:
        stride = window_size
    if stride <= 0:
        raise ValueError(f"stride > 0 필요. got {stride}")

    n_packets, n_sc = amplitude.shape
    if n_packets < window_size:
        if drop_last:
            return np.empty((0, window_size, n_sc), dtype=amplitude.dtype)
        padded = np.zeros((window_size, n_sc), dtype=amplitude.dtype)
        padded[:n_packets] = amplitude
        return padded[None, ...]

    n_windows = 1 + (n_packets - window_size) // stride
    starts = np.arange(n_windows) * stride
    out = np.stack([amplitude[s : s + window_size] for s in starts], axis=0)

    if not drop_last:
        last_end = starts[-1] + window_size
        if last_end < n_packets:
            tail = np.zeros((window_size, n_sc), dtype=amplitude.dtype)
            remain = n_packets - last_end
            tail[:remain] = amplitude[last_end:]
            out = np.concatenate([out, tail[None, ...]], axis=0)
    return out
