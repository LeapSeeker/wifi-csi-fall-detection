# server/receiver/udp_receiver.py

import socket
import struct
import threading
from typing import Optional
from config.settings import UDP_HOST, UDP_PORT

HEADER_FORMAT = "<BIQH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def parse_packet(raw: bytes) -> Optional[dict]:
    try:
        if len(raw) < HEADER_SIZE:
            return None

        device_id, seq_num, timestamp_us, n_subcarriers = struct.unpack_from(HEADER_FORMAT, raw, 0)

        amp_format = f"<{n_subcarriers}f"
        amp_size = struct.calcsize(amp_format)

        if len(raw) < HEADER_SIZE + amp_size:
            return None

        amplitudes = struct.unpack_from(amp_format, raw, HEADER_SIZE)

        return {
            "device_id": device_id,
            "seq_num": seq_num,
            "timestamp_us": timestamp_us,
            "n_subcarriers": n_subcarriers,
            "amplitudes": list(amplitudes)
        }
    except struct.error:
        return None

def start_receivers(callback):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    print(f"[UDP] 포트 {UDP_PORT} 수신 대기 중... (RX1/RX2 통합)")

    def listen():
        while True:
            raw, addr = sock.recvfrom(4096)
            packet = parse_packet(raw)
            if packet:
                callback(packet)

    t = threading.Thread(target=listen, daemon=True)
    t.start()