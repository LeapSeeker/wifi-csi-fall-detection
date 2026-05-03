# server/config/settings.py

# UDP 수신 설정 (ESP32 → 추론 서버)
UDP_HOST = "0.0.0.0"
UDP_PORT = 5005        # RX1, RX2 동일 포트 사용

# device_id 값
DEVICE_ID_RX1 = 0x01
DEVICE_ID_RX2 = 0x02

# WebSocket 설정 (Pi4 → 추론 서버)
WS_HOST = "0.0.0.0"
WS_PORT = 8765         # Pi4가 연결해오는 WebSocket 포트

# 페어링 버퍼 설정
PAIRING_TIMEOUT = 0.5  # 0.5초 안에 짝 못 찾으면 버림
BUFFER_MAX_SIZE = 50

# 서브캐리어 수
SUBCARRIER_COUNT = 64

# 서버 IP (시연/상용화 전환용)
# SERVER_IP = "123.456.789.0"  # 상용화: 공인 IP
SERVER_IP = "192.168.0.10"     # 시연: 핫스팟 로컬 IP (시연 당일 변경)