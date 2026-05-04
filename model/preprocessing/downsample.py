"""Alsaify(320Hz) -> ESP32 배포 환경(100Hz) 다운샘플링.

비율 100/320 = 5/16 -> resample_poly(up=5, down=16).
ESP32 샘플링 레이트와 정합되도록 사전학습 데이터를 맞춘다.
근거: model/CLAUDE.md "다운샘플링 근거" 절.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

UP = 5
DOWN = 16


def downsample_alsaify(amplitude: np.ndarray, axis: int = 0) -> np.ndarray:
    """320Hz -> 100Hz 다운샘플링.

    Parameters
    ----------
    amplitude : np.ndarray, shape (n_packets, n_subcarriers)
        진폭(real-valued) 배열.
    axis : int
        시간 축. 기본 0.

    Returns
    -------
    np.ndarray, float32
        시간 축 길이 = ceil(n_packets * 5 / 16).
    """
    out = resample_poly(amplitude, up=UP, down=DOWN, axis=axis)
    return out.astype(np.float32, copy=False)
