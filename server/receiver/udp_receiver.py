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
STATUS_MAGIC_BYTE = 0xAC
HEADER_FORMAT = "<BBbBIQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 16
AMP_FORMAT = f"<{SUBCARRIER_COUNT}f"
AMP_SIZE = struct.calcsize(AMP_FORMAT)         # 208
PACKET_SIZE = HEADER_SIZE + AMP_SIZE            # 224
STATUS_FORMAT = "<BB2xQIIIII"
STATUS_SIZE = struct.calcsize(STATUS_FORMAT)    # 32

_VALID_DEVICE_IDS = (DEVICE_ID_RX1, DEVICE_ID_RX2)


def parse_packet(raw: bytes) -> Optional[dict]:
    # D-007 패킷은 정확히 224B. 초과/부족 모두 거부 — 길이 mismatch는 펌웨어/네트워크 이상 신호.
    if len(raw) != PACKET_SIZE:
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


def parse_status_packet(raw: bytes) -> Optional[dict]:
    if len(raw) != STATUS_SIZE:
        return None
    try:
        magic, device_id, timestamp_us, cb, match, qfull, sent, fail = (
            struct.unpack_from(STATUS_FORMAT, raw, 0)
        )
    except struct.error:
        return None

    if magic != STATUS_MAGIC_BYTE:
        return None
    if device_id not in _VALID_DEVICE_IDS:
        return None

    return {
        "device_id": device_id,
        "timestamp_us": timestamp_us,
        "cb": cb,
        "match": match,
        "qfull": qfull,
        "sent": sent,
        "fail": fail,
    }


def _device_name(device_id: int) -> str:
    if device_id == DEVICE_ID_RX1:
        return "RX1"
    if device_id == DEVICE_ID_RX2:
        return "RX2"
    return f"0x{device_id:02X}"


def start_receivers(callback, port: Optional[int] = None):
    """UDP 수신 daemon thread 시작.

    bind 실패 시 RuntimeError 발생 (raw OSError traceback 노출 방지).
    호출 측은 RuntimeError를 잡아 사용자 친화적 메시지를 로그한다.
    """
    bind_port = UDP_PORT if port is None else port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 26214400)
    try:
        sock.bind((UDP_HOST, bind_port))
    except OSError as e:
        sock.close()
        raise RuntimeError(
            f"UDP 포트 {bind_port}에 바인드할 수 없습니다. "
            "collect 프로그램, 기존 서버 프로세스, "
            "또는 다른 수신기가 이미 실행 중인지 확인하세요."
        ) from e
    print(f"[UDP] 포트 {bind_port} 수신 대기 중... (RX1/RX2 통합)")

    def listen():
        while True:
            raw, _addr = sock.recvfrom(4096)
            if not raw:
                continue
            if raw[0] == STATUS_MAGIC_BYTE:
                status = parse_status_packet(raw)
                if status:
                    print(
                        "[STATUS] "
                        f"device={_device_name(status['device_id'])} "
                        f"cb={status['cb']} "
                        f"match={status['match']} "
                        f"qfull={status['qfull']} "
                        f"sent={status['sent']} "
                        f"fail={status['fail']}"
                    )
                continue

            packet = parse_packet(raw)
            if packet:
                callback(packet)

    t = threading.Thread(target=listen, daemon=True)
    t.start()
