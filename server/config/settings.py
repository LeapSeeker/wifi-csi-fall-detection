# server/config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

# 모드 설정 (demo / production)
MODE = os.getenv("MODE", "demo")

# 서버 IP 자동 선택
if MODE == "production":
    SERVER_IP = os.getenv("SERVER_IP_PRODUCTION", "123.456.789.0")
else:
    SERVER_IP = os.getenv("SERVER_IP_DEMO", "192.168.137.1")

print(f"[CONFIG] 모드: {MODE} | 서버 IP: {SERVER_IP}")

# UDP 수신 설정
UDP_HOST = "0.0.0.0"
UDP_PORT = 5005

# device_id 값
DEVICE_ID_RX1 = 0x01
DEVICE_ID_RX2 = 0x02

# WebSocket 설정
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# 페어링 버퍼 설정
PAIRING_TIMEOUT = 0.5
BUFFER_MAX_SIZE = 50

# 서브캐리어 수 (D-007: Rx 단일 패킷 LLTF 52)
SUBCARRIER_COUNT = 52

# Rx1+Rx2 concat 입력 서브캐리어 수 (D-013: 추론 입력 (300, 104))
SUBCARRIER_COUNT_CONCAT = 104