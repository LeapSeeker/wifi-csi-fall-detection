# server/utils/pairing.py

import time
import threading
from config.settings import PAIRING_TIMEOUT, BUFFER_MAX_SIZE

class PairingBuffer:
    def __init__(self, on_paired):
        """
        on_paired: 페어링 완료 시 호출될 콜백 함수
                   인자로 (rx1_packet, rx2_packet) 전달
        """
        self.buffer = {}        # seq → {rx_id: packet}
        self.lock = threading.Lock()
        self.on_paired = on_paired

    def add(self, packet: dict):
        seq = packet["seq"]
        rx_id = packet["rx_id"]

        with self.lock:
            # 버퍼 크기 초과 시 가장 오래된 항목 제거
            if len(self.buffer) >= BUFFER_MAX_SIZE:
                oldest_seq = min(self.buffer.keys())
                del self.buffer[oldest_seq]

            # seq 없으면 새로 만들기
            if seq not in self.buffer:
                self.buffer[seq] = {"time": time.time()}

            self.buffer[seq][rx_id] = packet

            # RX1(0)이랑 RX2(1) 둘 다 들어왔으면 페어링 완료
            if 0 in self.buffer[seq] and 1 in self.buffer[seq]:
                rx1 = self.buffer[seq][0]
                rx2 = self.buffer[seq][1]
                del self.buffer[seq]
                self.on_paired(rx1, rx2)

    def cleanup_expired(self):
        """타임아웃된 미완성 페어 제거 - 주기적으로 호출"""
        now = time.time()
        with self.lock:
            expired = [
                seq for seq, data in self.buffer.items()
                if now - data["time"] > PAIRING_TIMEOUT
            ]
            for seq in expired:
                del self.buffer[seq]