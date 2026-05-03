# server/utils/pairing.py

import time
import threading
from config.settings import PAIRING_TIMEOUT, BUFFER_MAX_SIZE, DEVICE_ID_RX1, DEVICE_ID_RX2

class PairingBuffer:
    def __init__(self, on_paired):
        self.buffer = {}
        self.lock = threading.Lock()
        self.on_paired = on_paired

    def add(self, packet: dict):
        seq = packet["seq_num"]
        device_id = packet["device_id"]

        with self.lock:
            if len(self.buffer) >= BUFFER_MAX_SIZE:
                oldest_seq = min(self.buffer.keys())
                del self.buffer[oldest_seq]

            if seq not in self.buffer:
                self.buffer[seq] = {"time": time.time()}

            self.buffer[seq][device_id] = packet

            if DEVICE_ID_RX1 in self.buffer[seq] and DEVICE_ID_RX2 in self.buffer[seq]:
                rx1 = self.buffer[seq][DEVICE_ID_RX1]
                rx2 = self.buffer[seq][DEVICE_ID_RX2]
                del self.buffer[seq]
                self.on_paired(rx1, rx2)

    def cleanup_expired(self):
        now = time.time()
        with self.lock:
            expired = [
                seq for seq, data in self.buffer.items()
                if now - data["time"] > PAIRING_TIMEOUT
            ]
            for seq in expired:
                del self.buffer[seq]