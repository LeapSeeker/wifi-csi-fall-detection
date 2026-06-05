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
FALL_THRESHOLD: float = 0.30  # 단일 모델 기준. 앙상블 사용 시 0.60~0.64

# 이벤트 확정 연속 윈도우 수 (FallEventConfirmer).
# N=1: 첫 positive 윈도우에서 즉시 발화 (시연 권장 — 지연 최소).
# N=2: 2연속 positive 확인 후 발화 (stride=100 기준 +1s 지연, 오탐 감소).
# diag_event_sweep.py 결과에서 최적값으로 교체할 것.
N_CONFIRM: int = 1

# 서브캐리어 수 (Rx1/Rx2 각각)
N_SUBCARRIERS_EACH: int = 52

# 실시간 추론 입력 시간축 정규화 (D-018 후속)
TARGET_SAMPLE_RATE_HZ: float = 100.0
RESAMPLE_MAX_GAP_MS: float = 100.0

# 모델 경로 — server/inference/config.py 기준으로 project_root 계산
# parents[0]=inference, parents[1]=server, parents[2]=project_root
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# [사전학습 데이터만 적용된 모델] Alsaify LOS 6-class. 자체수집 데이터 미반영.
#   Alsaify 테스트셋 기준 fall_recall 0.92 / FAR 0.03 (실 ESP32 도메인 갭 존재)
# MODEL_PATH: Path = _PROJECT_ROOT / "model" / "pretrained" / "checkpoints" / "best.pt"

# [최종 파인튜닝 모델] ss_peak160 + aug_merged(raw+sdp_shift) + recall_first + sr=0.9
#   단일 모델 (threshold=0.30): R=0.912 / FAR=0.075 / F1=0.835  → R,FAR 목표 달성
#   앙상블 f1f+rcf_sr90 w=0.5 (threshold=0.60): R=0.861 / FAR=0.043 / F1=0.853  → 전 항목 달성
MODEL_PATH: Path = (
    _PROJECT_ROOT
    / "model" / "finetune" / "checkpoints_rcf_sdp_shift_sr90" / "best_operating.pt"
)

# 앙상블 2nd 모델 (선택적, 병렬 추론 후 softmax 평균)
MODEL_PATH_2: Path = (
    _PROJECT_ROOT
    / "model" / "finetune" / "checkpoints_f1f_sdp_shift_t15" / "best_operating.pt"
)
