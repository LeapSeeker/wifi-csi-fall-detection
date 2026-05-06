# server/utils/pairing.py

import time
import threading
from config.settings import PAIRING_TIMEOUT, BUFFER_MAX_SIZE, DEVICE_ID_RX1, DEVICE_ID_RX2

class PairingBuffer:
    def __init__(self, on_paired):
        self.buffer = []        # [(timestamp_us, device_id, packet), ...]
        self.lock = threading.Lock()
        self.on_paired = on_paired

    def add(self, packet: dict):
        device_id = packet["device_id"]
        ts = packet["timestamp_us"]

        with self.lock:
            # 버퍼 크기 초과 시 가장 오래된 항목 제거
            if len(self.buffer) >= BUFFER_MAX_SIZE:
                self.buffer.pop(0)

            # 같은 시간대 (50ms 이내) 반대쪽 패킷 찾기
            WINDOW_US = 50_000  # 50ms
            for i, (other_ts, other_id, other_pkt) in enumerate(self.buffer):
                if other_id != device_id and abs(ts - other_ts) <= WINDOW_US:
                    # 페어링 완료
                    self.buffer.pop(i)
                    if device_id == DEVICE_ID_RX1:
                        rx1, rx2 = packet, other_pkt
                    else:
                        rx1, rx2 = other_pkt, packet
                    self.on_paired(rx1, rx2)
                    return

            # 짝 못 찾으면 버퍼에 추가
            self.buffer.append((ts, device_id, packet))

    def cleanup_expired(self):
        """타임아웃된 미완성 패킷 제거"""
        now_us = int(time.time() * 1_000_000)
        TIMEOUT_US = int(PAIRING_TIMEOUT * 1_000_000)

        with self.lock:
            self.buffer = [
                (ts, did, pkt) for ts, did, pkt in self.buffer
                if now_us - ts <= TIMEOUT_US
            ]