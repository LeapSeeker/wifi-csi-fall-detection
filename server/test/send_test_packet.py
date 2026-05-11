# server/test/send_test_packet.py

import socket
import struct
import time
import random

# 서버 설정
SERVER_IP = "127.0.0.1"
UDP_PORT = 5005          # 노션 설계 기준 단일 포트

# D-007 패킷 구조 (224B):
#   magic(1B,0xAB) | device_id(1B) | rssi(1B,int8_t) | reserved(1B)
#   | seq(4B) | timestamp_us(8B) | amplitude × 52 (float32, 208B)
MAGIC_BYTE = 0xAB
DEVICE_ID_RX1 = 0x01
DEVICE_ID_RX2 = 0x02

N_SUBCARRIERS = 52       # LLTF 유효 서브캐리어 수
HEADER_FORMAT = "<BBbBIQ"
AMP_FORMAT = f"<{N_SUBCARRIERS}f"
PACKET_SIZE = struct.calcsize(HEADER_FORMAT) + struct.calcsize(AMP_FORMAT)  # 224
assert PACKET_SIZE == 224, f"PACKET_SIZE {PACKET_SIZE} != 224"


def make_packet(device_id: int, seq_num: int, rssi: int, timestamp_us: int) -> bytes:
    amplitudes = [random.uniform(0.0, 1.0) for _ in range(N_SUBCARRIERS)]

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC_BYTE,
        device_id,
        rssi,
        0,
        seq_num,
        timestamp_us,
    )
    payload = struct.pack(AMP_FORMAT, *amplitudes)
    packet = header + payload
    assert len(packet) == 224, f"packet size {len(packet)} != 224"
    return packet


def send_test_packets(count: int = 20):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[TEST] 테스트 패킷 {count}개 전송 시작 → {SERVER_IP}:{UDP_PORT}")

    for seq in range(count):
        # RX1 전송
        ts_rx1 = int(time.time() * 1_000_000)
        packet_rx1 = make_packet(DEVICE_ID_RX1, seq, rssi=-45, timestamp_us=ts_rx1)
        sock.sendto(packet_rx1, (SERVER_IP, UDP_PORT))

        # 약간 딜레이 후 RX2 전송 (실제 환경 시뮬레이션)
        time.sleep(0.02)

        # RX2 전송 — timestamp_us 재측정 (Rx별 SNTP 타임스탬프 시뮬레이션)
        ts_rx2 = int(time.time() * 1_000_000)
        packet_rx2 = make_packet(DEVICE_ID_RX2, seq, rssi=-48, timestamp_us=ts_rx2)
        sock.sendto(packet_rx2, (SERVER_IP, UDP_PORT))

        print(f"[TEST] seq={seq} RX1/RX2 전송 완료")
        time.sleep(0.1)   # 10Hz 시뮬레이션

    print("[TEST] 전송 완료")
    sock.close()


if __name__ == "__main__":
    send_test_packets()
