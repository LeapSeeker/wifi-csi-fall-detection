# server/config/settings.py

# UDP 수신 설정
UDP_HOST = "0.0.0.0"       # 모든 네트워크 인터페이스에서 수신
UDP_PORT_RX1 = 5001        # ESP32 RX1 수신 포트
UDP_PORT_RX2 = 5002        # ESP32 RX2 수신 포트

# TCP 설정 (RPi4가 먼저 연결해옴)
TCP_HOST = "0.0.0.0"       # 서버가 모든 인터페이스에서 대기
TCP_PORT = 6000            # RPi4가 접속할 포트

# 페어링 버퍼 설정
PAIRING_TIMEOUT = 0.5      # 0.5초 안에 짝 못 찾으면 버림
BUFFER_MAX_SIZE = 50       # 버퍼 최대 프레임 수

# 서브캐리어 수
SUBCARRIER_COUNT = 64