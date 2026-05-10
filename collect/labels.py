"""SafeSignal 자체 데이터 수집 — 활동 코드 / 단계 / 수집 목표 정의.

자체 수집 목표 (270 세션):
- 낙상 9종 × 10회 = 90 세션
- 비낙상 6종 × 30회 = 180 세션
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
}


ACTIVITY_INFO: dict[str, dict] = {
    # 낙상 9종 — class_idx=0, target=10
    "FALL_SIT_F": {
        "display": "앉다→낙상 (앞)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "앉기", "duration": 2},
            {"name": "앞으로 낙상", "duration": 3},
        ],
    },
    "FALL_SIT_B": {
        "display": "앉다→낙상 (뒤)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "앉기", "duration": 2},
            {"name": "뒤로 낙상", "duration": 3},
        ],
    },
    "FALL_SIT_S": {
        "display": "앉다→낙상 (옆)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "앉기", "duration": 2},
            {"name": "옆으로 낙상", "duration": 3},
        ],
    },
    "FALL_STD_F": {
        "display": "서다→낙상 (앞)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "서있기", "duration": 2},
            {"name": "앞으로 낙상", "duration": 3},
        ],
    },
    "FALL_STD_B": {
        "display": "서다→낙상 (뒤)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "서있기", "duration": 2},
            {"name": "뒤로 낙상", "duration": 3},
        ],
    },
    "FALL_STD_S": {
        "display": "서다→낙상 (옆)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "서있기", "duration": 2},
            {"name": "옆으로 낙상", "duration": 3},
        ],
    },
    "FALL_WALK_F": {
        "display": "걷다→낙상 (앞)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "걷기", "duration": 2},
            {"name": "앞으로 낙상", "duration": 3},
        ],
    },
    "FALL_WALK_B": {
        "display": "걷다→낙상 (뒤)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "걷기", "duration": 2},
            {"name": "뒤로 낙상", "duration": 3},
        ],
    },
    "FALL_WALK_S": {
        "display": "걷다→낙상 (옆)",
        "class_idx": 0,
        "target": 10,
        "stages": [
            {"name": "걷기", "duration": 2},
            {"name": "옆으로 낙상", "duration": 3},
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
