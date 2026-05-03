# server/receiver/udp_receiver.py

import socket
import struct
import threading
from typing import Optional
from config.settings import UDP_HOST, UDP_PORT_RX1, UDP_PORT_RX2, SUBCARRIER_COUNT

PACKET_FORMAT = "<BBIfd" + f"{SUBCARRIER_COUNT}f"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

def parse_packet(raw: bytes) -> Optional[dict]:
    try:
        unpacked = struct.unpack(PACKET_FORMAT, raw[:PACKET_SIZE])
        magic, rx_id, seq, timestamp, rssi, subcarrier_count, *amplitudes = unpacked
        if magic != 0xAB:
            return None
        return {
            "rx_id": rx_id,
            "seq": seq,
            "timestamp": timestamp,
            "rssi": rssi,
            "amplitudes": list(amplitudes)
        }
    except struct.error:
        return None

def start_udp_listener(port: int, callback):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, port))
    print(f"[UDP] 포트 {port} 수신 대기 중...")

    while True:
        raw, addr = sock.recvfrom(4096)
        packet = parse_packet(raw)
        if packet:
            callback(packet)

def start_receivers(callback):
    for port in [UDP_PORT_RX1, UDP_PORT_RX2]:
        t = threading.Thread(
            target=start_udp_listener,
            args=(port, callback),
            daemon=True
        )
        t.start()