"""z-score 적용 *전* SDP energy 계산 — predictor energy gate와 calibrate 도구가
공유하는 단일 helper.

지시서 보강 요구: predictor의 gate와 calibration이 각자 따로 energy를 계산하면
metric이 갈라져 캘리브레이션값이 gate에서 다르게 동작한다. 반드시 한 곳에 정의하고
양쪽이 import 한다.

energy 정의는 `debug/preprocessing/analyze_sdp_energy.py`와 동일하다:
    rpca_sparse → stacked_doppler_profile → (z-score 적용 직전) metric 계산.
stacked_doppler_profile에 전달하는 sub_w/stride/n_lags 는 함수 기본값과 동일하므로
analyze_sdp_energy.py(기본값 호출)와 같은 SDP를 산출한다.

torch 비의존(numpy + 전처리 모듈만). calibrate 도구가 torch 로드 없이 import 가능.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# project_root 경로 추가 → model.preprocessing.* import
# parents[0]=inference, parents[1]=server, parents[2]=project_root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.preprocessing.acf import N_LAGS  # noqa: E402
from model.preprocessing.rpca import DEFAULT_MAX_ITER, rpca_sparse  # noqa: E402
from model.preprocessing.sdp import (  # noqa: E402
    SUB_STRIDE,
    SUB_W,
    stacked_doppler_profile,
)

# z-score 전 SDP energy metric 후보 (출력/검증 순서 고정)
ENERGY_METRICS: tuple[str, ...] = (
    "sdp_mean_abs",
    "sdp_fro",
    "sdp_std",
    "sdp_max_abs",
    "sparse_ratio",
    "raw_std",
    "raw_delta_mean",
)


def _sdp_metrics(sdp: np.ndarray, sparse: np.ndarray, window: np.ndarray) -> dict[str, float]:
    """z-score 전 SDP/raw energy metric dict. analyze_sdp_energy._window_metrics와 동일식."""
    abs_sdp = np.abs(sdp)
    win_fro = float(np.linalg.norm(window))
    sparse_fro = float(np.linalg.norm(sparse))
    return {
        "sdp_mean_abs": float(abs_sdp.mean()),
        "sdp_fro": float(np.linalg.norm(sdp)),
        "sdp_std": float(sdp.std()),
        "sdp_max_abs": float(abs_sdp.max()),
        "sparse_ratio": float(sparse_fro / win_fro) if win_fro > 0 else 0.0,
        "raw_std": float(window.std()),
        "raw_delta_mean": float(np.abs(np.diff(window, axis=0)).mean())
        if window.shape[0] >= 2 else 0.0,
    }


def compute_window_energy(
    window: np.ndarray,
    rpca_max_iter: int = DEFAULT_MAX_ITER,
    rpca_tol: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """(n_t, n_sc) window → (z-score 전 SDP, energy metric dict).

    predictor는 반환된 SDP로 z-score를 적용해 모델 입력을 만들고(RPCA 1회 재사용),
    calibrate 도구는 metric dict만 사용한다. 양쪽이 이 함수를 호출하므로 동일 window
    입력에서 동일 energy 값이 나온다.
    """
    sparse = rpca_sparse(window, max_iter=rpca_max_iter, tol=rpca_tol)
    sdp = stacked_doppler_profile(
        sparse, sub_w=SUB_W, stride=SUB_STRIDE, n_lags=N_LAGS
    )
    metrics = _sdp_metrics(sdp, sparse, window)
    return sdp, metrics
