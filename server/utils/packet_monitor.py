# server/utils/packet_monitor.py

import threading

class PacketMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_seq = {0x01: None, 0x02: None}
        self.total_received = {0x01: 0, 0x02: 0}
        self.total_lost = {0x01: 0, 0x02: 0}

    def update(self, packet: dict):
        device_id = packet["device_id"]
        seq = packet["seq_num"]

        with self.lock:
            self.total_received[device_id] += 1
            if self.last_seq[device_id] is not None:
                expected = self.last_seq[device_id] + 1
                if seq > expected:
                    self.total_lost[device_id] += seq - expected
            self.last_seq[device_id] = seq

    def _loss_rate(self, device_id: int) -> float:
        """락 없이 손실률 계산 (내부용)"""
        total = self.total_received[device_id] + self.total_lost[device_id]
        if total == 0:
            return 0.0
        return round(self.total_lost[device_id] / total * 100, 2)

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "rx1": {
                    "received": self.total_received[0x01],
                    "lost": self.total_lost[0x01],
                    "loss_rate": self._loss_rate(0x01)
                },
                "rx2": {
                    "received": self.total_received[0x02],
                    "lost": self.total_lost[0x02],
                    "loss_rate": self._loss_rate(0x02)
                }
            }

    def reset(self):
        with self.lock:
            self.last_seq = {0x01: None, 0x02: None}
            self.total_received = {0x01: 0, 0x02: 0}
            self.total_lost = {0x01: 0, 0x02: 0}