# server/utils/cooldown.py

import time
import threading

class FallCooldown:
    def __init__(self, cooldown_sec: int = 30):
        """
        cooldown_sec: 낙상 감지 후 다음 알림까지 대기 시간 (기본 30초)
        """
        self.cooldown_sec = cooldown_sec
        self.last_fall_time = 0
        self.lock = threading.Lock()

    def check_and_update(self) -> dict:
        """낙상 알림 발송 가능 여부와 cooldown 상태를 함께 반환.

        allowed=True일 때만 last_fall_time을 현재 시각으로 갱신한다.
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_fall_time
            if elapsed >= self.cooldown_sec:
                self.last_fall_time = now
                return {
                    "allowed": True,
                    "now": now,
                    "elapsed": elapsed,
                    "remaining": 0.0,
                    "cooldown_sec": self.cooldown_sec,
                }

            remaining = max(0.0, self.cooldown_sec - elapsed)
            print(f"[COOLDOWN] 쿨다운 중... {int(remaining)}초 후 알림 가능")
            return {
                "allowed": False,
                "now": now,
                "elapsed": elapsed,
                "remaining": remaining,
                "cooldown_sec": self.cooldown_sec,
            }

    def is_allowed(self) -> bool:
        """낙상 알림 발송 가능 여부 확인."""
        return bool(self.check_and_update()["allowed"])

    def reset(self):
        """쿨다운 수동 초기화 (오탐 취소 버튼용)"""
        with self.lock:
            self.last_fall_time = 0
            print("[COOLDOWN] 쿨다운 초기화 완료")
