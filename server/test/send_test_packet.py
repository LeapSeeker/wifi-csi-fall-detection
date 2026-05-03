# server/test/send_test_packet.py

import socket
import struct
import time
import random

# 서버 설정
SERVER_IP = "127.0.0.1"   # 같은 노트북에서 테스트
UDP_PORT_RX1 = 5001
UDP_PORT_RX2 = 5002

SUBCARRIER_COUNT = 64
PACKET_FORMAT = "<BBIfd" + f"{SUBCARRIER_COUNT}f"

def make_packet(rx_id: int, seq: int) -> bytes:
    magic = 0xAB
    timestamp = time.time()
    rssi = random.uniform(-70.0, -30.0)
    amplitudes = [random.uniform(0.0, 1.0) for _ in range(SUBCARRIER_COUNT)]

    return struct.pack(
        PACKET_FORMAT,
        magic, rx_id, seq, timestamp, rssi, *amplitudes
    )

def send_test_packets(count: int = 20):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[TEST] 테스트 패킷 {count}개 전송 시작")

    for seq in range(count):
        # RX1 전송
        packet_rx1 = make_packet(rx_id=0, seq=seq)
        sock.sendto(packet_rx1, (SERVER_IP, UDP_PORT_RX1))

        # 약간 딜레이 후 RX2 전송 (실제 환경 시뮬레이션)
        time.sleep(0.02)

        # RX2 전송
        packet_rx2 = make_packet(rx_id=1, seq=seq)
        sock.sendto(packet_rx2, (SERVER_IP, UDP_PORT_RX2))

        print(f"[TEST] seq={seq} 전송 완료")
        time.sleep(0.1)   # 10Hz 시뮬레이션

    print("[TEST] 전송 완료")
    sock.close()

if __name__ == "__main__":
    send_test_packets()