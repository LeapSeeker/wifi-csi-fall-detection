"""추론 파이프라인 상수 (D-013/D-014/D-015).

MODEL_PATH 만 동적 계산. 나머지는 단순 상수.
"""
from __future__ import annotations

from pathlib import Path

# 윈도우/슬라이딩 (D-014: stride=100 기본, RTX4060 실측 후 조정 예정)
WINDOW_SIZE: int = 300
INFERENCE_STRIDE: int = 100

# 낙상 클래스. pretrained 6-class는 "fall", fine-tuned 7-class는 방향별 fall_* 가능.
FALL_LABELS: tuple[str, ...] = (
    "fall",
    "fall_forward",
    "fall_backward",
    "fall_side",
)
FALL_THRESHOLD: float = 0.5

# 서브캐리어 수 (Rx1/Rx2 각각)
N_SUBCARRIERS_EACH: int = 52

# 실시간 추론 입력 시간축 정규화 (D-018 후속)
TARGET_SAMPLE_RATE_HZ: float = 100.0
RESAMPLE_MAX_GAP_MS: float = 100.0

# best.pt 경로 — server/inference/config.py 기준으로 project_root 계산
# parents[0]=inference, parents[1]=server, parents[2]=project_root
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODEL_PATH: Path = _PROJECT_ROOT / "model" / "pretrained" / "checkpoints" / "best.pt"
