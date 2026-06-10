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

    def is_allowed(self) -> bool:
        """낙상 알림 발송 가능 여부 확인"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_fall_time
            if elapsed >= self.cooldown_sec:
                self.last_fall_time = now
                return True
            else:
                remaining = int(self.cooldown_sec - elapsed)
                print(f"[COOLDOWN] 쿨다운 중... {remaining}초 후 알림 가능")
                return False

    def reset(self):
        """쿨다운 수동 초기화 (오탐 취소 버튼용)"""
        with self.lock:
            self.last_fall_time = 0
            print("[COOLDOWN] 쿨다운 초기화 완료")