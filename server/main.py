# server/main.py

import os
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
from collect_manager import CollectManager
from dotenv import load_dotenv

# -----------------------------------------------
# 전역 핸들 (Windows multiprocessing 안전: top-level에는 인스턴스 X)
# -----------------------------------------------
rpi_connection: RPiConnection = None
fall_cooldown: FallCooldown = None
packet_monitor: PacketMonitor = None
pairing_buffer: PairingBuffer = None
inference_worker: InferenceWorker = None
collect_manager: CollectManager = None
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
        return

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


def record_activity_result(result: dict):
    """추후 1주일 단위 생활 패턴 분석용 일반 행동 저장 지점."""
    # 저장 형식(JSONL/CSV/SQLite 등)은 아직 미정이다.
    return


def handle_inference_result(result: dict):
    """InferenceWorker 결과를 낙상 알림과 일반 행동 저장 흐름으로 분기."""
    if result.get("is_fall"):
        on_fall_detected(
            confidence=result.get("confidence", 0.0),
            seq_num=result.get("seq_num", 0),
            timestamp_us=result.get("timestamp_us", 0),
        )
        return

    record_activity_result(result)

# -----------------------------------------------
# 페어링 완료 시 호출되는 콜백
# -----------------------------------------------
def on_paired(rx1, rx2):
    last_fall_pair["rx1"] = rx1
    last_fall_pair["rx2"] = rx2
    log_pair(rx1, rx2)

    # is_recording을 한 번만 읽어 분기 기준을 고정한다.
    # (수집/추론 동시 전달 또는 양쪽 누락 방지)
    is_collecting = collect_manager is not None and collect_manager.is_recording

    # 수집 모드: collect_manager에 먼저 반영해야 update_pair가 emit하는
    # collect_pair_count가 현재 수집된 페어 수와 일치한다 (1개 지연 방지).
    if is_collecting:
        collect_manager.add_pair(rx1, rx2)

    update_pair(rx1, rx2)

    # 추론: 수집 중이 아닐 때만 실행
    if inference_worker is not None and not is_collecting:
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
    """InferenceWorker 결과 폴링 → handle_inference_result()로 분기."""
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
        handle_inference_result(result)


def main():
    global rpi_connection, fall_cooldown, packet_monitor, pairing_buffer, inference_worker, collect_manager

    rpi_connection = RPiConnection(on_status_change=update_rpi4_status)
    load_dotenv()  # .env에서 환경 변수 로드
    fall_cooldown = FallCooldown(cooldown_sec=int(os.getenv("COOLDOWN_SEC", "30")))
    packet_monitor = PacketMonitor()
    pairing_buffer = PairingBuffer(on_paired=on_paired)
    inference_worker = InferenceWorker()
    collect_manager = CollectManager()

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
    start_dashboard(on_fall_detected=on_fall_detected, collect_manager=collect_manager)


# -----------------------------------------------
# 메인 실행
# -----------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main() or 0)
