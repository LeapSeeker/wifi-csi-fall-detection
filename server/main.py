# server/main.py

import multiprocessing
import sys
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
from inference.worker import InferenceWorker

# -----------------------------------------------
# 전역 핸들 (Windows multiprocessing 안전: top-level에는 인스턴스 X)
# -----------------------------------------------
rpi_connection: RPiConnection = None
fall_cooldown: FallCooldown = None
packet_monitor: PacketMonitor = None
pairing_buffer: PairingBuffer = None
inference_worker: InferenceWorker = None
fall_count: int = 0
last_fall_pair = {"rx1": None, "rx2": None}

# -----------------------------------------------
# 낙상 감지 시 실행 (InferenceWorker 결과 → 알림)
# -----------------------------------------------
def on_fall_detected(
    confidence: float = 0.0,
    seq_num: int = 0,
    timestamp_us: int = 0,
):
    global fall_count

    if rpi_connection is None or fall_cooldown is None:
        return  # 초기화 전 콜백 호출 방어

    if not fall_cooldown.is_allowed():
        log_warn("낙상 감지됐지만 쿨다운 중 — 알림 생략")
        return

    fall_count += 1
    log_fall(fall_count)
    update_fall()
    rpi_connection.send_fall_alert(
        confidence=confidence,
        seq_num=seq_num,
        timestamp_us=timestamp_us,
    )
    send_fall_sms()

    if last_fall_pair["rx1"] and last_fall_pair["rx2"]:
        save_fall(fall_count, last_fall_pair["rx1"], last_fall_pair["rx2"])

# -----------------------------------------------
# 페어링 완료 시 호출되는 콜백 → InferenceWorker로 비동기 전달
# -----------------------------------------------
def on_paired(rx1, rx2):
    last_fall_pair["rx1"] = rx1
    last_fall_pair["rx2"] = rx2
    log_pair(rx1, rx2)
    update_pair(rx1, rx2)

    if inference_worker is not None:
        inference_worker.put(rx1, rx2)

# -----------------------------------------------
# UDP 수신 콜백
# -----------------------------------------------
def on_packet_received(packet):
    if packet_monitor is None or pairing_buffer is None:
        return
    packet_monitor.update(packet)
    pairing_buffer.add(packet)

# -----------------------------------------------
# 백그라운드 루프
# -----------------------------------------------
def stats_loop():
    while True:
        if packet_monitor is not None:
            stats = packet_monitor.get_stats()
            update_packet_stats(stats)
        time.sleep(5)


def cleanup_loop():
    while True:
        if pairing_buffer is not None:
            pairing_buffer.cleanup_expired()
        time.sleep(1)


def result_loop():
    """InferenceWorker 결과 폴링 → 낙상이면 on_fall_detected() 호출."""
    while True:
        if inference_worker is None:
            time.sleep(0.1)
            continue
        result = inference_worker.get_result()
        if result is None:
            time.sleep(0.05)
            continue
        if "error" in result:
            log_warn(f"[InferenceWorker] {result['error']}")
            continue
        if result.get("is_fall"):
            on_fall_detected(
                confidence=result.get("confidence", 0.0),
                seq_num=result.get("seq_num", 0),
                timestamp_us=result.get("timestamp_us", 0),
            )


def main():
    global rpi_connection, fall_cooldown, packet_monitor, pairing_buffer, inference_worker

    rpi_connection = RPiConnection(on_status_change=update_rpi4_status)
    fall_cooldown = FallCooldown(cooldown_sec=30)
    packet_monitor = PacketMonitor()
    pairing_buffer = PairingBuffer(on_paired=on_paired)
    inference_worker = InferenceWorker()

    log_info("서버 시작")
    log_info(f"로그 파일: {get_log_filepath()}")

    rpi_connection.start()
    log_info("WebSocket 서버 시작 완료")

    inference_worker.start()
    log_info("InferenceWorker 시작 완료")

    try:
        start_receivers(callback=on_packet_received)
    except RuntimeError as e:
        log_warn(str(e))
        return 1
    log_info("UDP 수신 시작 완료")

    threading.Thread(target=cleanup_loop, daemon=True).start()
    threading.Thread(target=stats_loop, daemon=True).start()
    threading.Thread(target=result_loop, daemon=True).start()

    log_info("대시보드: http://localhost:8080")
    start_dashboard()


# -----------------------------------------------
# 메인 실행
# -----------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main() or 0)
