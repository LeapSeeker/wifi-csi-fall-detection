# server/main.py

import time
import threading
from receiver.udp_receiver import start_receivers
from utils.pairing import PairingBuffer
from tcp_handler.rpi_connection import RPiConnection
from notification.sms import send_fall_sms

# -----------------------------------------------
# RPi4 TCP 연결 인스턴스
# -----------------------------------------------
rpi_connection = RPiConnection()

# -----------------------------------------------
# 낙상 감지 시 실행
# -----------------------------------------------
def on_fall_detected():
    print("[FALL] 낙상 감지!")
    rpi_connection.send_fall_alert()   # RPi4에 TCP 알림
    send_fall_sms()                    # 보호자에게 SMS 발송

# -----------------------------------------------
# 페어링 완료 시 호출되는 콜백
# -----------------------------------------------
def on_paired(rx1, rx2):
    print(f"[PAIR] seq={rx1['seq']} | "
          f"RX1 rssi={rx1['rssi']:.1f} | "
          f"RX2 rssi={rx2['rssi']:.1f}")

    # TODO: AI 추론 팀 모듈 연결 (전처리 → CNN-LSTM)
    # result = inference(rx1, rx2)
    # if result == "fall":
    #     on_fall_detected()

# -----------------------------------------------
# 메인 실행
# -----------------------------------------------
if __name__ == "__main__":
    print("[SERVER] 서버 시작")

    # TCP 서버 시작 (RPi4 연결 대기)
    rpi_connection.start()

    # 페어링 버퍼 초기화
    pairing_buffer = PairingBuffer(on_paired=on_paired)

    # UDP 수신 시작 (RX1, RX2 각각 스레드로 실행)
    start_receivers(callback=pairing_buffer.add)

    # 타임아웃된 페어 주기적으로 정리 (1초마다)
    def cleanup_loop():
        while True:
            pairing_buffer.cleanup_expired()
            time.sleep(1)

    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

    print("[SERVER] UDP 수신 대기 중... (종료: Ctrl+C)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[SERVER] 서버 종료")