# server/receiver/udp_receiver.py

import socket
import struct
import threading
from typing import Optional
from config.settings import (
    UDP_HOST,
    UDP_PORT,
    SUBCARRIER_COUNT,
    DEVICE_ID_RX1,
    DEVICE_ID_RX2,
)

# D-007 패킷 구조 (224B):
#   magic(1B,0xAB) | device_id(1B) | rssi(1B,int8_t) | reserved(1B)
#   | seq(4B) | timestamp_us(8B) | amplitude × 52 (float32, 208B)
MAGIC_BYTE = 0xAB
HEADER_FORMAT = "<BBbBIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 16
AMP_FORMAT = f"<{SUBCARRIER_COUNT}f"
AMP_SIZE = struct.calcsize(AMP_FORMAT)         # 208
PACKET_SIZE = HEADER_SIZE + AMP_SIZE            # 224

_VALID_DEVICE_IDS = (DEVICE_ID_RX1, DEVICE_ID_RX2)


def parse_packet(raw: bytes) -> Optional[dict]:
    if len(raw) < PACKET_SIZE:
        return None
    try:
        magic, device_id, rssi, _reserved, seq_num, timestamp_us = struct.unpack_from(
            HEADER_FORMAT, raw, 0
        )
    except struct.error:
        return None

    if magic != MAGIC_BYTE:
        return None
    if device_id not in _VALID_DEVICE_IDS:
        return None

    try:
        amplitudes = struct.unpack_from(AMP_FORMAT, raw, HEADER_SIZE)
    except struct.error:
        return None

    return {
        "device_id": device_id,
        "rssi": rssi,
        "seq_num": seq_num,
        "timestamp_us": timestamp_us,
        "n_subcarriers": SUBCARRIER_COUNT,
        "amplitudes": list(amplitudes),
    }


def start_receivers(callback):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    print(f"[UDP] 포트 {UDP_PORT} 수신 대기 중... (RX1/RX2 통합)")

    def listen():
        while True:
            raw, _addr = sock.recvfrom(4096)
            packet = parse_packet(raw)
            if packet:
                callback(packet)

    t = threading.Thread(target=listen, daemon=True)
    t.start()
