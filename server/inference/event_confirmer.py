"""이벤트 레벨 낙상 확정 후처리.

윈도우 단위 fall 확률을 받아 N 연속 확정 시 이벤트를 발화한다.
stride=100(1s) 기준 N=1 → 첫 positive 윈도우에서 즉시 발화.
N=2 → 2 연속 positive 확인 후 발화 (1초 추가 지연, 오탐 감소).
"""
from __future__ import annotations

from collections import deque


class FallEventConfirmer:
    """윈도우 fall 확률 → N 연속 확정 → 이벤트 발화.

    사용:
        confirmer = FallEventConfirmer(threshold=0.25, n_confirm=1)

        # result_loop 에서 매 윈도우 결과마다 호출
        if confirmer.push(result["confidence"], result.get("is_fall", False)):
            on_fall_detected(...)
            confirmer.reset()   # 낙상 알림 후 버퍼 초기화
    """

    def __init__(self, threshold: float, n_confirm: int = 1) -> None:
        if n_confirm < 1:
            raise ValueError(f"n_confirm must be >= 1, got {n_confirm}")
        self.threshold = threshold
        self.n_confirm = n_confirm
        self._window: deque[bool] = deque(maxlen=n_confirm)

    def push(self, fall_prob: float, is_fall: bool | None = None) -> bool:
        """새 윈도우 결과를 받아 이벤트 확정 여부를 반환.

        fall_prob 이 threshold 이상이거나 is_fall=True 이면 positive.
        N 연속 positive 가 쌓이면 True 반환.
        """
        positive = (fall_prob >= self.threshold) or bool(is_fall)
        self._window.append(positive)
        if len(self._window) < self.n_confirm:
            return False
        return all(self._window)

    def reset(self) -> None:
        """낙상 알림 발송 후 버퍼 초기화."""
        self._window.clear()

    @property
    def pending(self) -> int:
        """현재 버퍼에 쌓인 윈도우 수."""
        return len(self._window)
