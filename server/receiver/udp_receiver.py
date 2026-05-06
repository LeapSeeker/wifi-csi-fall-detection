# server/receiver/udp_receiver.py

import socket
import struct
import threading
from typing import Optional
from config.settings import UDP_HOST, UDP_PORT

# 펌웨어 패킷 구조 (224 bytes)
# magic(1) | device_id(1) | rssi(1) | reserved(1) | seq(4) | timestamp_us(8) | amplitude(52×4)
PACKET_MAGIC = 0xAB
SUBCARRIER_COUNT = 52
HEADER_FORMAT = "<BBbBIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
PACKET_SIZE = HEADER_SIZE + SUBCARRIER_COUNT * 4  # 224 bytes

def parse_packet(raw: bytes) -> Optional[dict]:
    try:
        if len(raw) < PACKET_SIZE:
            print(f"[DEBUG] 크기 부족: {len(raw)} < {PACKET_SIZE}")
            return None

        magic, device_id, rssi, reserved, seq, timestamp_us = struct.unpack_from(HEADER_FORMAT, raw, 0)
        print(f"[DEBUG] magic={magic:02X}, device_id={device_id:02X}, rssi={rssi}, seq={seq}")

        if magic != PACKET_MAGIC:
            print(f"[DEBUG] magic 불일치: {magic:02X} != {PACKET_MAGIC:02X}")
            return None

        amp_format = f"<{SUBCARRIER_COUNT}f"
        amplitudes = struct.unpack_from(amp_format, raw, HEADER_SIZE)

        print(f"[DEBUG] 파싱 성공!")
        return {
            "device_id": device_id,
            "seq_num": seq,
            "timestamp_us": timestamp_us,
            "rssi": rssi,
            "n_subcarriers": SUBCARRIER_COUNT,
            "amplitudes": list(amplitudes)
        }
    except struct.error as e:
        print(f"[DEBUG] struct 오류: {e}")
        return None

def start_receivers(callback):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    print(f"[UDP] 포트 {UDP_PORT} 수신 대기 중... (RX1/RX2 통합)")

    def listen():
        while True:
            raw, addr = sock.recvfrom(4096)
            print(f"[DEBUG] 패킷 수신: {len(raw)} bytes, magic={raw[0]:02X}, device_id={raw[1]:02X}")
            packet = parse_packet(raw)
            if packet:
                callback(packet)
            else:
                print(f"[DEBUG] 파싱 실패")

    t = threading.Thread(target=listen, daemon=True)
    t.start()