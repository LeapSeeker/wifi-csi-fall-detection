"""300 패킷(=3초 @100Hz) 슬라이딩 윈도우 분절."""
from __future__ import annotations

import numpy as np

WINDOW_SIZE = 300  # 패킷 (3초 @100Hz, SHARED.md 모델 입력 스펙)


def sliding_windows(
    amplitude: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    drop_last: bool = True,
    *,
    tail_window: bool = False,
    pad_short: bool = False,
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
    tail_window : bool, keyword-only
        True면 마지막 정방향 윈도우 이후 잔여 패킷이 있을 때
        amplitude[-window_size:] 슬라이스를 윈도우 1개 추가한다.
        잔여가 0이면 아무것도 추가하지 않는다. drop_last=False와
        동시에 True면 tail_window가 우선하여 zero-padding 대신
        overlap 윈도우를 추가한다. 기본값은 False.
    pad_short : bool, keyword-only
        True면 n_packets < window_size 인 경우에도 zero-padding을 적용하여
        윈도우 1개를 생성한다. padding은 데이터 뒷부분에 추가된다.
        drop_last 값과 무관하게 우선 적용된다. 기본값은 False.

    Returns
    -------
    np.ndarray, shape (n_windows, window_size, n_subcarriers)
        n_packets < window_size 인 경우:
          - pad_short=True 이면 zero-padding(뒷부분)된 윈도우 1개를 담아
            (1, window_size, n_subcarriers) 반환 (drop_last 무관 우선).
          - pad_short=False, drop_last=True 이면 empty (0, window_size, n_subcarriers).
          - pad_short=False, drop_last=False 이면 zero-padding(뒷부분) 윈도우 1개.
        n_packets >= window_size 인 경우:
          정방향 슬라이딩으로 만든 윈도우들을 반환하며, tail_window=True 이고
          마지막 윈도우 이후 잔여가 있으면 amplitude[-window_size:] 윈도우
          1개가 추가되어 (n_windows+1, ...) 가 된다.
    """
    if amplitude.ndim != 2:
        raise ValueError(f"2D 입력 필요 (n_packets, n_sc). got shape={amplitude.shape}")
    if stride is None:
        stride = window_size
    if stride <= 0:
        raise ValueError(f"stride > 0 필요. got {stride}")

    n_packets, n_sc = amplitude.shape
    if n_packets < window_size:
        if pad_short:
            padded = np.zeros((window_size, n_sc), dtype=amplitude.dtype)
            padded[:n_packets] = amplitude
            return padded[None, ...]
        if drop_last:
            return np.empty((0, window_size, n_sc), dtype=amplitude.dtype)
        padded = np.zeros((window_size, n_sc), dtype=amplitude.dtype)
        padded[:n_packets] = amplitude
        return padded[None, ...]

    n_windows = 1 + (n_packets - window_size) // stride
    starts = np.arange(n_windows) * stride
    out = np.stack([amplitude[s : s + window_size] for s in starts], axis=0)

    last_end = starts[-1] + window_size
    if tail_window:
        if last_end < n_packets:
            tail = amplitude[-window_size:]
            out = np.concatenate([out, tail[None, ...]], axis=0)
    elif not drop_last:
        if last_end < n_packets:
            tail = np.zeros((window_size, n_sc), dtype=amplitude.dtype)
            remain = n_packets - last_end
            tail[:remain] = amplitude[last_end:]
            out = np.concatenate([out, tail[None, ...]], axis=0)
    return out
