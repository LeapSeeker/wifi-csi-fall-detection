# server/receiver/udp_receiver.py

import socket
import struct
import threading
from typing import Optional
from config.settings import UDP_HOST, UDP_PORT, SUBCARRIER_COUNT

# 페이로드 구조: device_id(1) | seq_num(4) | timestamp_us(8) | n_subcarriers(2) | CSI amplitude(N×4)
# magic, rssi 없음 (노션 설계 기준)
HEADER_FORMAT = "<BIH"   # device_id(1) + seq_num(4) + timestamp_us(8) + n_subcarriers(2)
# timestamp_us는 8바이트 unsigned long long
HEADER_FORMAT = "<BIQH"  # B=device_id, I=seq_num, Q=timestamp_us, H=n_subcarriers
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def parse_packet(raw: bytes) -> Optional[dict]:
    try:
        if len(raw) < HEADER_SIZE:
            return None

        device_id, seq_num, timestamp_us, n_subcarriers = struct.unpack_from(HEADER_FORMAT, raw, 0)

        # CSI amplitude 파싱
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