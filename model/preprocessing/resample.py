"""SafeSignal 자체수집 CSV의 100Hz 균일 격자 리샘플 (D-018).

자체수집 환경(Windows 모바일 핫스팟 + ESP32 promiscuous)에서 페어 rate가
~70Hz 천장에 막혀 있어, 모델 사전학습 100Hz 가정과 시간축이 어긋난다.
이 모듈은 timestamp 기반으로 100Hz(=10,000us 간격) 균일 격자를 생성하고
각 서브캐리어를 선형 보간하여 모델 입력 시간축을 정규화한다.

원칙:
  - 정보 복원이 아니라 시간축 정규화.
  - 외삽 금지(첫/마지막 timestamp 안쪽만 채움).
  - 큰 timestamp gap은 hard reject 하지 않고 ResampleResult.metadata로 남긴다.
  - 선형 보간만 사용(np.interp). cubic/spline은 overshoot 위험으로 미사용.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResampleResult:
    amplitude: np.ndarray          # (n_resampled, n_sc) float32
    timestamps_us: np.ndarray      # (n_resampled,) int64
    original_count: int            # 입력 패킷 수 (정렬/중복 정리 전)
    resampled_count: int           # 출력 균일 격자 길이
    original_rate_hz: float        # (n-1) / duration_s, 0이면 NaN
    target_hz: float
    max_gap_us: int                # 입력 timestamp의 최대 인접 간격
    gap_count: int                 # max_gap_ms 초과 gap 개수


def resample_to_100hz(
    amp: np.ndarray,
    timestamps_us: np.ndarray,
    target_hz: float = 100.0,
    max_gap_ms: float = 100.0,
) -> ResampleResult:
    """timestamp 기반 균일 격자(100Hz) 선형 보간.

    Parameters
    ----------
    amp : np.ndarray, shape (n_packets, n_subcarriers), float
    timestamps_us : np.ndarray, shape (n_packets,), int/float (마이크로초)
    target_hz : float
        목표 샘플레이트. 100.0이면 step=10,000us.
    max_gap_ms : float
        품질 통계용 임계값. 이 값을 초과하는 입력 timestamp 인접 간격을
        gap_count로 집계한다. hard reject는 하지 않는다.

    동작
    ----
    1) (timestamp, amp)를 timestamp 기준 stable sort
    2) 동일 timestamp 중복은 평균값으로 병합
       (중복은 펌웨어 cb 재진입/페어링 동시 도착에서 드물게 발생; 정보 손실
       최소화 위해 평균. drop이 아닌 이유는 같은 시점 두 측정값을 모두 무시하지
       않기 위함.)
    3) 단조 증가 검증
    4) 첫 timestamp ~ 마지막 timestamp 사이를 step=1e6/target_hz us 간격의
       균일 격자로 만들어 각 subcarrier별 np.interp 선형 보간
    5) 외삽 금지: 균일 격자 끝점은 마지막 입력 timestamp 이하

    Returns
    -------
    ResampleResult
    """
    if amp.ndim != 2:
        raise ValueError(f"amp는 2D (n_packets, n_sc) 이어야 함. got {amp.shape}")
    timestamps_us = np.asarray(timestamps_us)
    if timestamps_us.ndim != 1 or timestamps_us.shape[0] != amp.shape[0]:
        raise ValueError(
            f"timestamps_us shape {timestamps_us.shape}가 amp[0] {amp.shape[0]}와 불일치"
        )

    original_count = int(amp.shape[0])
    n_sc = amp.shape[1]
    if target_hz <= 0:
        raise ValueError(f"target_hz는 0보다 커야 함. got {target_hz}")
    step_us = int(round(1_000_000.0 / target_hz))
    if step_us <= 0:
        raise ValueError(f"target_hz={target_hz}가 너무 커서 step_us={step_us}")

    if original_count < 2:
        # 보간 불가 — 빈 결과 반환, metadata로 남김
        return ResampleResult(
            amplitude=np.empty((0, n_sc), dtype=np.float32),
            timestamps_us=np.empty((0,), dtype=np.int64),
            original_count=original_count,
            resampled_count=0,
            original_rate_hz=float("nan"),
            target_hz=target_hz,
            max_gap_us=0,
            gap_count=0,
        )

    # 1) stable sort by timestamp (timestamp 역전 안전 처리)
    order = np.argsort(timestamps_us, kind="stable")
    ts_sorted = timestamps_us[order].astype(np.int64, copy=False)
    amp_sorted = amp[order].astype(np.float32, copy=False)

    # 2) 동일 timestamp 중복 → 평균 병합
    #    np.unique(return_inverse=True) + np.add.reduceat로 그룹 평균.
    uniq_ts, inverse = np.unique(ts_sorted, return_inverse=True)
    if uniq_ts.shape[0] != ts_sorted.shape[0]:
        sums = np.zeros((uniq_ts.shape[0], n_sc), dtype=np.float64)
        counts = np.zeros((uniq_ts.shape[0],), dtype=np.int64)
        np.add.at(sums, inverse, amp_sorted)
        np.add.at(counts, inverse, 1)
        amp_unique = (sums / counts[:, None]).astype(np.float32)
        ts_unique = uniq_ts
    else:
        amp_unique = amp_sorted
        ts_unique = uniq_ts

    # 3) 단조 증가 검증 (정렬 후이므로 진단용)
    diffs = np.diff(ts_unique)
    if np.any(diffs <= 0):
        # 동일 타임스탬프는 위에서 병합됐으므로 여기 도달은 비정상
        raise RuntimeError(
            "timestamp가 단조 증가하지 않음 (정렬/병합 후): "
            f"min diff={diffs.min()}"
        )
    max_gap_us = int(diffs.max()) if diffs.size > 0 else 0
    gap_count = int(np.sum(diffs > max_gap_ms * 1000.0))

    # original_rate_hz: (n-1) / duration_s
    duration_s = (ts_unique[-1] - ts_unique[0]) / 1_000_000.0
    if duration_s > 0:
        original_rate_hz = float((ts_unique.shape[0] - 1) / duration_s)
    else:
        original_rate_hz = float("nan")

    # 4-5) 균일 격자 (외삽 금지). step_us 간격, 첫 ts 이상, 마지막 ts 이하.
    start_us = int(ts_unique[0])
    end_us = int(ts_unique[-1])
    n_grid = (end_us - start_us) // step_us + 1
    if n_grid <= 0:
        return ResampleResult(
            amplitude=np.empty((0, n_sc), dtype=np.float32),
            timestamps_us=np.empty((0,), dtype=np.int64),
            original_count=original_count,
            resampled_count=0,
            original_rate_hz=original_rate_hz,
            target_hz=target_hz,
            max_gap_us=max_gap_us,
            gap_count=gap_count,
        )
    grid_us = start_us + np.arange(n_grid, dtype=np.int64) * step_us

    # subcarrier별 np.interp (np.interp는 외삽을 두 끝점 값으로 clamp 하지만
    # grid가 [start, end] 범위 안이므로 외삽이 발생하지 않는다.)
    out = np.empty((n_grid, n_sc), dtype=np.float32)
    xp = ts_unique.astype(np.float64)
    for j in range(n_sc):
        out[:, j] = np.interp(
            grid_us.astype(np.float64), xp, amp_unique[:, j].astype(np.float64)
        ).astype(np.float32)

    return ResampleResult(
        amplitude=out,
        timestamps_us=grid_us,
        original_count=original_count,
        resampled_count=int(n_grid),
        original_rate_hz=original_rate_hz,
        target_hz=target_hz,
        max_gap_us=max_gap_us,
        gap_count=gap_count,
    )
