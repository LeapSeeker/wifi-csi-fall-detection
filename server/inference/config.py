"""추론 파이프라인 설정 (D-013/D-014/D-015).

기본값은 2026-06-10 데모 최종 운영값이다. 환경변수로 override 가능하다.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    return int(val)


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    return float(val)


# 윈도우/슬라이딩 (데모 최종: stride=150, drop 0 확인)
WINDOW_SIZE: int = 300
INFERENCE_STRIDE: int = _env_int("SAFESIGNAL_INFERENCE_STRIDE", 150)

# 낙상 클래스. pretrained 6-class는 "fall", fine-tuned 7-class는 방향별 fall_* 가능.
FALL_LABELS: tuple[str, ...] = (
    "fall",
    "fall_forward",
    "fall_backward",
    "fall_side",
)
FALL_THRESHOLD: float = _env_float("SAFESIGNAL_FALL_THRESHOLD", 0.30)

# 연속 fire 판정. 1이면 기존 동작과 동일.
FALL_CONSECUTIVE_N: int = max(1, _env_int("SAFESIGNAL_FALL_CONSECUTIVE_N", 1))

# SDP energy hard gate. 기본 off라 기존 E2E 동작과 동일.
ENERGY_GATE_ENABLED: bool = _env_bool("SAFESIGNAL_ENERGY_GATE_ENABLED", False)
ENERGY_GATE_THRESHOLD: float = _env_float("SAFESIGNAL_ENERGY_GATE_THRESHOLD", 0.0)
ENERGY_GATE_METRIC: str = os.getenv("SAFESIGNAL_ENERGY_GATE_METRIC", "sdp_mean_abs").strip()

# RPCA 반복 횟수. 데모 최종: 100 (p95 latency 3279ms -> 1478ms).
RPCA_MAX_ITER: int = _env_int("SAFESIGNAL_RPCA_MAX_ITER", 100)

# 서브캐리어 수 (Rx1/Rx2 각각)
N_SUBCARRIERS_EACH: int = 52

# 실시간 추론 입력 시간축 정규화 (D-018 후속)
TARGET_SAMPLE_RATE_HZ: float = 100.0
RESAMPLE_MAX_GAP_MS: float = 100.0

# 모델 경로 — server/inference/config.py 기준으로 project_root 계산
# parents[0]=inference, parents[1]=server, parents[2]=project_root
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _resolve_model_path() -> Path:
    raw = os.getenv("SAFESIGNAL_MODEL_PATH", "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path.resolve()
    return (
        _PROJECT_ROOT
        / "model" / "finetune" / "checkpoints_item4_balanced"
        / "onset_primary_balanced_aug_s42" / "best_operating.pt"
    )

# [사전학습 데이터만 적용된 모델] Alsaify LOS 6-class. 자체수집 데이터 미반영.
#   Alsaify 테스트셋 기준 fall_recall 0.92 / FAR 0.03 (실 ESP32 도메인 갭 존재)
# MODEL_PATH: Path = _PROJECT_ROOT / "model" / "pretrained" / "checkpoints" / "best.pt"

# [데모 최종 모델] onset_primary_balanced_aug_s42
#   validation 기준 fall_recall 0.808 / FAR 0.010 / F1 0.824 (threshold=0.30)
MODEL_PATH: Path = _resolve_model_path()
