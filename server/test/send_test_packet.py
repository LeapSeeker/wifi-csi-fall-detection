# server/test/send_test_packet.py

import socket
import struct
import time
import random

# 서버 설정
SERVER_IP = "127.0.0.1"
UDP_PORT = 5005          # 노션 설계 기준 단일 포트

# device_id
DEVICE_ID_RX1 = 0x01
DEVICE_ID_RX2 = 0x02

# 페이로드 구조: device_id(1) | seq_num(4) | timestamp_us(8) | n_subcarriers(2) | CSI amplitude(N×4)
N_SUBCARRIERS = 52       # LLTF 유효 서브캐리어 수
HEADER_FORMAT = "<BIQH"
AMP_FORMAT = f"<{N_SUBCARRIERS}f"

def make_packet(device_id: int, seq_num: int) -> bytes:
    timestamp_us = int(time.time() * 1_000_000)
    n_subcarriers = N_SUBCARRIERS
    amplitudes = [random.uniform(0.0, 1.0) for _ in range(N_SUBCARRIERS)]

    header = struct.pack(HEADER_FORMAT, device_id, seq_num, timestamp_us, n_subcarriers)
    payload = struct.pack(AMP_FORMAT, *amplitudes)
    return header + payload

def send_test_packets(count: int = 20):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[TEST] 테스트 패킷 {count}개 전송 시작 → {SERVER_IP}:{UDP_PORT}")

    for seq in range(count):
        # RX1 전송
        packet_rx1 = make_packet(DEVICE_ID_RX1, seq)
        sock.sendto(packet_rx1, (SERVER_IP, UDP_PORT))

        # 약간 딜레이 후 RX2 전송 (실제 환경 시뮬레이션)
        time.sleep(0.02)

        # RX2 전송
        packet_rx2 = make_packet(DEVICE_ID_RX2, seq)
        sock.sendto(packet_rx2, (SERVER_IP, UDP_PORT))

        print(f"[TEST] seq={seq} RX1/RX2 전송 완료")
        time.sleep(0.1)   # 10Hz 시뮬레이션

    print("[TEST] 전송 완료")
    sock.close()

if __name__ == "__main__":
    send_test_packets()