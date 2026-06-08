"""추론 파이프라인 설정 (D-013/D-014/D-015).

기본값은 기존 E2E 동작을 보존한다. 데모용 방어선은 환경변수로만 켠다.
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


# 윈도우/슬라이딩 (D-014: stride=100 기본, RTX4060 실측 후 조정 예정)
WINDOW_SIZE: int = 300
INFERENCE_STRIDE: int = _env_int("SAFESIGNAL_INFERENCE_STRIDE", 100)

# 낙상 클래스. pretrained 6-class는 "fall", fine-tuned 7-class는 방향별 fall_* 가능.
FALL_LABELS: tuple[str, ...] = (
    "fall",
    "fall_forward",
    "fall_backward",
    "fall_side",
)
FALL_THRESHOLD: float = _env_float("SAFESIGNAL_FALL_THRESHOLD", 0.5)

# 연속 fire 판정. 1이면 기존 동작과 동일.
FALL_CONSECUTIVE_N: int = max(1, _env_int("SAFESIGNAL_FALL_CONSECUTIVE_N", 1))

# SDP energy hard gate. 기본 off라 기존 E2E 동작과 동일.
ENERGY_GATE_ENABLED: bool = _env_bool("SAFESIGNAL_ENERGY_GATE_ENABLED", False)
ENERGY_GATE_THRESHOLD: float = _env_float("SAFESIGNAL_ENERGY_GATE_THRESHOLD", 0.0)
ENERGY_GATE_METRIC: str = os.getenv("SAFESIGNAL_ENERGY_GATE_METRIC", "sdp_mean_abs").strip()

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
        / "model" / "finetune" / "checkpoints_track1_formal_global_s43" / "best_operating.pt"
    )

# [사전학습 데이터만 적용된 모델] Alsaify LOS 6-class. 자체수집 데이터 미반영.
#   Alsaify 테스트셋 기준 fall_recall 0.92 / FAR 0.03 (실 ESP32 도메인 갭 존재)
# MODEL_PATH: Path = _PROJECT_ROOT / "model" / "pretrained" / "checkpoints" / "best.pt"

# [예비 학습 모델] 자체수집 ESP32 데이터 파인튜닝 (track1_formal_global_s43, 미확정 임시본)
#   within_subject 기준 fall_recall 0.778 / FAR 0.150 / F1 0.687 (threshold=0.1 측정)
MODEL_PATH: Path = _resolve_model_path()
