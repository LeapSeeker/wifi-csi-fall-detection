# server/main.py

import time
import threading
from receiver.udp_receiver import start_receivers
from utils.pairing import PairingBuffer
from utils.cooldown import FallCooldown
from utils.packet_monitor import PacketMonitor
from ws_handler.rpi_connection import RPiConnection
from notification.sms import send_fall_sms
from dashboard.app import start_dashboard, update_pair, update_fall, update_rpi4_status, update_packet_stats
from logger.log_manager import log_info, log_warn, log_pair, log_fall, get_log_filepath
from logger.fall_history import save_fall

# -----------------------------------------------
# 인스턴스 초기화
# -----------------------------------------------
rpi_connection = RPiConnection(on_status_change=update_rpi4_status)
fall_cooldown = FallCooldown(cooldown_sec=30)
packet_monitor = PacketMonitor()
fall_count = 0
last_fall_pair = {"rx1": None, "rx2": None}

# -----------------------------------------------
# 낙상 감지 시 실행
# -----------------------------------------------
def on_fall_detected():
    global fall_count

    if not fall_cooldown.is_allowed():
        log_warn("낙상 감지됐지만 쿨다운 중 — 알림 생략")
        return

    fall_count += 1
    log_fall(fall_count)
    update_fall()
    rpi_connection.send_fall_alert()
    send_fall_sms()

    if last_fall_pair["rx1"] and last_fall_pair["rx2"]:
        save_fall(fall_count, last_fall_pair["rx1"], last_fall_pair["rx2"])

# -----------------------------------------------
# 페어링 완료 시 호출되는 콜백
# -----------------------------------------------
def on_paired(rx1, rx2):
    last_fall_pair["rx1"] = rx1
    last_fall_pair["rx2"] = rx2
    log_pair(rx1, rx2)
    update_pair(rx1, rx2)

    # TODO: AI 추론 팀 모듈 연결
    # result = inference(rx1, rx2)
    # if result == "fall":
    #     on_fall_detected()

# -----------------------------------------------
# 페어링 버퍼 (전역으로 선언)
# -----------------------------------------------
pairing_buffer = PairingBuffer(on_paired=on_paired)

# -----------------------------------------------
# UDP 수신 콜백
# -----------------------------------------------
def on_packet_received(packet):
    packet_monitor.update(packet)
    pairing_buffer.add(packet)

# -----------------------------------------------
# 패킷 손실률 주기적 업데이트
# -----------------------------------------------
def stats_loop():
    while True:
        stats = packet_monitor.get_stats()
        update_packet_stats(stats)
        time.sleep(5)

# -----------------------------------------------
# 메인 실행
# -----------------------------------------------
if __name__ == "__main__":
    log_info("서버 시작")
    log_info(f"로그 파일: {get_log_filepath()}")

    rpi_connection.start()
    log_info("WebSocket 서버 시작 완료")

    start_receivers(callback=on_packet_received)
    log_info("UDP 수신 시작 완료")

    def cleanup_loop():
        while True:
            pairing_buffer.cleanup_expired()
            time.sleep(1)

    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()

    stats_thread = threading.Thread(target=stats_loop, daemon=True)
    stats_thread.start()

    log_info("대시보드: http://localhost:8080")
    start_dashboard()