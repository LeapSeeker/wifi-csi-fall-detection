# server/main.py

import time
import threading
from receiver.udp_receiver import start_receivers
from utils.pairing import PairingBuffer
from ws_handler.rpi_connection import RPiConnection
from notification.sms import send_fall_sms
from dashboard.app import start_dashboard, update_pair, update_fall, update_rpi4_status

# -----------------------------------------------
# Pi4 WebSocket 연결 인스턴스
# -----------------------------------------------
rpi_connection = RPiConnection(on_status_change=update_rpi4_status)

# -----------------------------------------------
# 낙상 감지 시 실행
# -----------------------------------------------
def on_fall_detected():
    print("[FALL] 낙상 감지!")
    update_fall()                          # 대시보드 업데이트
    rpi_connection.send_fall_alert()       # Pi4에 WebSocket 알림
    send_fall_sms()                        # 보호자에게 SMS 발송

# -----------------------------------------------
# 페어링 완료 시 호출되는 콜백
# -----------------------------------------------
def on_paired(rx1, rx2):
    print(f"[PAIR] seq={rx1['seq_num']} | "
          f"RX1 subs={rx1['n_subcarriers']} | "
          f"RX2 subs={rx2['n_subcarriers']}")
    update_pair(rx1, rx2)                  # 대시보드 업데이트

    # TODO: AI 추론 팀 모듈 연결 (RPCA → ACF → SDP → z-score → CNN+GRU+Attention)
    # result = inference(rx1, rx2)
    # if result == "fall":
    #     on_fall_detected()

# -----------------------------------------------
# 메인 실행
# -----------------------------------------------
if __name__ == "__main__":
    print("[SERVER] 서버 시작")

    # WebSocket 서버 시작 (Pi4 연결 대기)
    rpi_connection.start()

    # 페어링 버퍼 초기화
    pairing_buffer = PairingBuffer(on_paired=on_paired)

    # UDP 수신 시작
    start_receivers(callback=pairing_buffer.add)

    # 타임아웃된 페어 주기적으로 정리 (1초마다)
    def cleanup_loop():
        while True:
            pairing_buffer.cleanup_expired()
            time.sleep(1)

    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

    # 대시보드 시작 (http://localhost:8080)
    print("[SERVER] 대시보드: http://localhost:8080")
    start_dashboard()