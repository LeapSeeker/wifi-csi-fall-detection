"""SafeSignal 자체 데이터 수집 — 활동 코드 / 단계 / 수집 목표 정의.

자체 수집 목표 (242 세션):
- 낙상 6종 × 10회 = 60 세션
- 비낙상 6종 × 30회 = 180 세션
- no_motion baseline 2회

세션 수/target은 그대로이며, fall 세션의 stage 구성과 duration만 event 중심으로 변경됨.
각 fall 세션은 event 중심 4초 프로토콜(대기 1초 → 낙상 2초 → 낙상 후 정지 1초)로
실제 낙상 순간을 짧고 명확하게 포착한다.
"""

from __future__ import annotations

CLASS_NAMES: dict[int, str] = {
    0: "fall",
    1: "walking",
    2: "sit_stand",
    3: "lying",
    4: "standing",
    5: "running",
    6: "picking",
    7: "no_motion",
}


ACTIVITY_INFO: dict[str, dict] = {
    # 낙상 6종 — class_idx=0, target=10
    # side fall(FALL_*_S)은 fine-tuning 기본 수집/학습/평가 범위에서 제외.
    "FALL_SIT_F": {
        "display": "앉다→낙상 (앞)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "앉은 상태 대기", "duration": 1},
            {"name": "앞으로 낙상", "duration": 2},
            {"name": "낙상 후 정지", "duration": 1},
        ],
    },
    "FALL_SIT_B": {
        "display": "앉다→낙상 (뒤)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "앉은 상태 대기", "duration": 1},
            {"name": "뒤로 낙상", "duration": 2},
            {"name": "낙상 후 정지", "duration": 1},
        ],
    },
    "FALL_STD_F": {
        "display": "서다→낙상 (앞)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "선 상태 대기", "duration": 1},
            {"name": "앞으로 낙상", "duration": 2},
            {"name": "낙상 후 정지", "duration": 1},
        ],
    },
    "FALL_STD_B": {
        "display": "서다→낙상 (뒤)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "선 상태 대기", "duration": 1},
            {"name": "뒤로 낙상", "duration": 2},
            {"name": "낙상 후 정지", "duration": 1},
        ],
    },
    "FALL_WALK_F": {
        "display": "걷다→낙상 (앞)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "1~2걸음 걷기", "duration": 1},
            {"name": "앞으로 낙상", "duration": 2},
            {"name": "낙상 후 정지", "duration": 1},
        ],
    },
    "FALL_WALK_B": {
        "display": "걷다→낙상 (뒤)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "1~2걸음 걷기", "duration": 1},
            {"name": "뒤로 낙상", "duration": 2},
            {"name": "낙상 후 정지", "duration": 1},
        ],
    },
    # 비낙상 6종 — target=30
    "SIT_STD": {
        "display": "앉기/일어서기",
        "class_idx": 2,
        "target": 30,
        "stages": [
            {"name": "앉기", "duration": 3},
            {"name": "일어서기", "duration": 3},
        ],
    },
    "LIE": {
        "display": "눕기/일어서기",
        "class_idx": 3,
        "target": 30,
        "stages": [
            {"name": "눕기", "duration": 3},
            {"name": "일어서기", "duration": 4},
        ],
    },
    "WALK": {
        "display": "걷기",
        "class_idx": 1,
        "target": 30,
        "duration": 8,
    },
    "STAND": {
        "display": "서있기",
        "class_idx": 4,
        "target": 30,
        "duration": 5,
    },
    "RUN": {
        "display": "빠른 걷기",
        "class_idx": 5,
        "target": 30,
        "duration": 8,
    },
    "PICK": {
        "display": "물건 줍기",
        "class_idx": 6,
        "target": 30,
        "duration": 5,
    },
    "NO_MOTION": {
        "display": "무동작/빈 공간",
        "class_idx": 7,
        "target": 2,
        "duration": 300,
    },
}


ACTIVITY_ORDER: list[str] = list(ACTIVITY_INFO.keys())


def get_duration(activity_code: str) -> int:
    """활동 코드의 총 녹화 길이(초)를 반환한다."""
    info = ACTIVITY_INFO[activity_code]
    if "stages" in info:
        return sum(s["duration"] for s in info["stages"])
    return info["duration"]


def total_target_sessions() -> int:
    return sum(info["target"] for info in ACTIVITY_INFO.values())
