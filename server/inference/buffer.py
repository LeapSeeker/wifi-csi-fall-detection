"""SlidingWindowBuffer — Rx1+Rx2 concat 행렬을 100Hz 시간축 윈도우로 적재.

D-013: Rx1(52) + Rx2(52) → concat 104.
D-018 후속: 실시간 입력도 timestamp 기반 100Hz 균일 격자로 보간.
"""
from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from .config import (
    INFERENCE_STRIDE,
    N_SUBCARRIERS_EACH,
    RESAMPLE_MAX_GAP_MS,
    TARGET_SAMPLE_RATE_HZ,
    WINDOW_SIZE,
)


def _resample_uniform(
    amp: np.ndarray,
    timestamps_us: np.ndarray,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal timestamp-based linear resampling for realtime inference."""
    if amp.ndim != 2:
        raise ValueError(f"amp must be 2D. got {amp.shape}")
    if timestamps_us.ndim != 1 or timestamps_us.shape[0] != amp.shape[0]:
        raise ValueError("timestamps_us length must match amp rows")
    if amp.shape[0] < 2:
        return (
            np.empty((0, amp.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    order = np.argsort(timestamps_us, kind="stable")
    ts_sorted = timestamps_us[order].astype(np.int64, copy=False)
    amp_sorted = amp[order].astype(np.float32, copy=False)

    uniq_ts, inverse = np.unique(ts_sorted, return_inverse=True)
    if uniq_ts.shape[0] != ts_sorted.shape[0]:
        sums = np.zeros((uniq_ts.shape[0], amp.shape[1]), dtype=np.float64)
        counts = np.zeros((uniq_ts.shape[0],), dtype=np.int64)
        np.add.at(sums, inverse, amp_sorted)
        np.add.at(counts, inverse, 1)
        amp_unique = (sums / counts[:, None]).astype(np.float32)
        ts_unique = uniq_ts
    else:
        amp_unique = amp_sorted
        ts_unique = uniq_ts

    if ts_unique.shape[0] < 2:
        return (
            np.empty((0, amp.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    step_us = int(round(1_000_000.0 / target_hz))
    start_us = int(ts_unique[0])
    end_us = int(ts_unique[-1])
    n_grid = (end_us - start_us) // step_us + 1
    if n_grid <= 0:
        return (
            np.empty((0, amp.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    grid_us = start_us + np.arange(n_grid, dtype=np.int64) * step_us
    out = np.empty((n_grid, amp.shape[1]), dtype=np.float32)
    xp = ts_unique.astype(np.float64)
    grid_x = grid_us.astype(np.float64)
    for j in range(amp.shape[1]):
        out[:, j] = np.interp(
            grid_x, xp, amp_unique[:, j].astype(np.float64)
        ).astype(np.float32)
    return out, grid_us


class SlidingWindowBuffer:
    """Rx1/Rx2 페어를 timestamp 기반 100Hz 윈도우로 누적."""

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        stride: int = INFERENCE_STRIDE,
        n_sc_each: int = N_SUBCARRIERS_EACH,
        target_hz: float = TARGET_SAMPLE_RATE_HZ,
        max_gap_ms: float = RESAMPLE_MAX_GAP_MS,
    ):
        self.window_size = window_size
        self.stride = stride
        self.n_sc_each = n_sc_each
        self.n_cols = n_sc_each * 2
        self.target_hz = target_hz
        self.max_gap_ms = max_gap_ms
        self.step_us = int(round(1_000_000.0 / target_hz))
        if self.step_us <= 0:
            raise ValueError(f"invalid target_hz={target_hz}")

        # 100Hz 3초 윈도우를 만들기 위해 raw pair는 여유 있게 보존한다.
        self._raw_maxlen = max(window_size * 4, window_size + stride * 2)
        self._raw_rows: Deque[np.ndarray] = deque(maxlen=self._raw_maxlen)
        self._raw_ts_us: Deque[int] = deque(maxlen=self._raw_maxlen)
        self._window: np.ndarray | None = None
        self._window_timestamp_us: int = 0
        self._last_trigger_grid_ts_us: int | None = None
        self._synthetic_ts_us: int = 0

    @property
    def window_timestamp_us(self) -> int:
        """현재 윈도우의 마지막 100Hz grid timestamp."""
        return self._window_timestamp_us

    def add(self, rx1_amp, rx2_amp, timestamp_us: int | None = None) -> bool:
        """rx1/rx2 진폭 리스트(각 52)를 concat해 row로 추가.

        timestamp_us가 제공되면 Rx1 timestamp 기준으로 100Hz 균일 격자에
        리샘플한다. None/0이면 테스트 호환을 위해 100Hz synthetic timestamp를
        사용한다.

        Returns
        -------
        bool : 100Hz 윈도우가 가득 차고 stride 조건을 만족하면 True.
        """
        if len(rx1_amp) != self.n_sc_each or len(rx2_amp) != self.n_sc_each:
            return False
        row = np.concatenate(
            (np.asarray(rx1_amp, dtype=np.float32),
             np.asarray(rx2_amp, dtype=np.float32))
        )

        ts_us = int(timestamp_us or 0)
        if ts_us <= 0:
            self._synthetic_ts_us += self.step_us
            ts_us = self._synthetic_ts_us

        self._raw_rows.append(row)
        self._raw_ts_us.append(ts_us)

        if len(self._raw_rows) < 2:
            return False

        amp = np.stack(self._raw_rows, axis=0).astype(np.float32, copy=False)
        timestamps = np.asarray(self._raw_ts_us, dtype=np.int64)
        resampled_amp, resampled_ts = _resample_uniform(
            amp,
            timestamps,
            target_hz=self.target_hz,
        )
        if resampled_amp.shape[0] < self.window_size:
            return False

        grid_ts_us = int(resampled_ts[-1])
        trigger_interval_us = self.stride * self.step_us
        if (
            self._last_trigger_grid_ts_us is None
            or grid_ts_us - self._last_trigger_grid_ts_us >= trigger_interval_us
        ):
            self._window = resampled_amp[-self.window_size:].astype(
                np.float32, copy=False
            )
            self._window_timestamp_us = grid_ts_us
            self._last_trigger_grid_ts_us = grid_ts_us
            return True
        return False

    def get_window(self) -> np.ndarray:
        """(WINDOW_SIZE, 104) float32 행렬 반환."""
        if self._window is None:
            raise RuntimeError("window is not ready")
        return self._window.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self._raw_rows)
