"""SlidingWindowBuffer — Rx1+Rx2 concat 행렬을 윈도우 단위로 적재.

D-013: Rx1(52) + Rx2(52) → concat 104 / D-014: stride=100 기본.
"""
from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from .config import INFERENCE_STRIDE, N_SUBCARRIERS_EACH, WINDOW_SIZE


class SlidingWindowBuffer:
    """Rx1/Rx2 페어를 행 단위로 (WINDOW_SIZE, 104) deque에 누적."""

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        stride: int = INFERENCE_STRIDE,
        n_sc_each: int = N_SUBCARRIERS_EACH,
    ):
        self.window_size = window_size
        self.stride = stride
        self.n_sc_each = n_sc_each
        self.n_cols = n_sc_each * 2
        self._buf: Deque[np.ndarray] = deque(maxlen=window_size)
        # stride 와 동일 값으로 초기화 → 버퍼가 처음 가득 차는 순간 즉시 trigger.
        self._since_last_predict = stride

    def add(self, rx1_amp, rx2_amp) -> bool:
        """rx1/rx2 진폭 리스트(각 52)를 concat해 row로 추가.

        Returns
        -------
        bool : 윈도우가 가득 차고 stride 조건을 만족하면 True (추론 트리거).
        """
        if len(rx1_amp) != self.n_sc_each or len(rx2_amp) != self.n_sc_each:
            return False
        row = np.concatenate(
            (np.asarray(rx1_amp, dtype=np.float32),
             np.asarray(rx2_amp, dtype=np.float32))
        )
        self._buf.append(row)

        if len(self._buf) < self.window_size:
            return False

        self._since_last_predict += 1
        if self._since_last_predict >= self.stride:
            self._since_last_predict = 0
            return True
        return False

    def get_window(self) -> np.ndarray:
        """(WINDOW_SIZE, 104) float32 행렬 반환."""
        return np.stack(self._buf, axis=0).astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self._buf)
