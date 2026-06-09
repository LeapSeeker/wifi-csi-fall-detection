# server/main.py

import os
import multiprocessing
import sys
import time
import threading
import datetime

from receiver.udp_receiver import start_receivers
from utils.pairing import PairingBuffer
from utils.cooldown import FallCooldown
from utils.packet_monitor import PacketMonitor
from ws_handler.rpi_connection import RPiConnection
from notification.sms import send_fall_sms
from dashboard.app import start_dashboard, update_pair, update_fall, update_rpi4_status, update_packet_stats, emit_rpi_log
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
    log_fall(fall_count, confidence=confidence)
    update_fall()
    rpi_connection.send_fall_alert(
        confidence=confidence,
        seq_num=seq_num,
        timestamp_us=timestamp_us,
    )
    send_fall_sms()

    if last_fall_pair["rx1"] and last_fall_pair["rx2"]:
        save_fall(fall_count, last_fall_pair["rx1"], last_fall_pair["rx2"])


def _is_inference_disabled() -> bool:
    """SAFESIGNAL_DISABLE_INFERENCE 환경변수 평가 (load_dotenv 이후 호출).

    허용 truthy 값: "1", "true", "yes", "on" (대소문자 무관). 그 외/미설정은 False.
    """
    val = os.getenv("SAFESIGNAL_DISABLE_INFERENCE", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def record_activity_result(result: dict):
    """추후 1주일 단위 생활 패턴 분석용 일반 행동 저장 지점."""
    # 저장 형식(JSONL/CSV/SQLite 등)은 아직 미정이다.
    return


# -----------------------------------------------
# [임시 진단] 매 윈도우 fall_confidence/energy/timestamp 추적 (SAFESIGNAL_CONF_DIAG=1)
#   오탐 원인 분석용. 정상 운영에는 영향 없음 (env 미설정 시 no-op). 분석 후 제거 예정.
# -----------------------------------------------
_conf_diag_fp = None
_conf_diag_prev_win_ts = {"v": None}


def _conf_diag_init():
    global _conf_diag_fp
    val = os.getenv("SAFESIGNAL_CONF_DIAG", "").strip().lower()
    if val not in ("1", "true", "yes", "on"):
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(os.path.dirname(__file__), "logs", f"conf_diag_{ts}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _conf_diag_fp = open(path, "a", encoding="utf-8")
    _conf_diag_fp.write(
        "wall,seq,win_ts_us,d_ts_us,fall_conf,top_conf,top_class,"
        "raw_is_fall,is_fall,consec,sdp_mean_abs,raw_delta_mean\n"
    )
    _conf_diag_fp.flush()
    log_info(f"[conf-diag] 매 윈도우 추론값 기록: {path}")


def _conf_diag_write(result: dict):
    if _conf_diag_fp is None:
        return
    win_ts = int(result.get("timestamp_us", 0))
    prev = _conf_diag_prev_win_ts["v"]
    d_ts = (win_ts - prev) if prev is not None else 0
    _conf_diag_prev_win_ts["v"] = win_ts
    metrics = (result.get("energy_gate") or {}).get("metrics") or {}
    consec = (result.get("fall_consecutive") or {}).get("count", 0)
    _conf_diag_fp.write(
        f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]},"
        f"{result.get('seq_num', 0)},{win_ts},{d_ts},"
        f"{result.get('fall_confidence', 0.0):.4f},"
        f"{result.get('confidence', 0.0):.4f},"
        f"{result.get('class', '')},"
        f"{int(bool(result.get('raw_is_fall')))},{int(bool(result.get('is_fall')))},"
        f"{consec},"
        f"{metrics.get('sdp_mean_abs', float('nan')):.6f},"
        f"{metrics.get('raw_delta_mean', float('nan')):.6f}\n"
    )
    _conf_diag_fp.flush()


def handle_inference_result(result: dict):
    """InferenceWorker 결과를 낙상 알림과 일반 행동 저장 흐름으로 분기."""
    _conf_diag_write(result)  # [임시 진단] 매 윈도우 raw 추론값 기록
    if result.get("is_fall"):
        # 0-1순위: 판정식이 fall_confidence >= threshold 로 정렬되면서 argmax class가
        # non-fall 인데도 is_fall=True 가 될 수 있다. fall 알림/Pi4/SMS confidence 는
        # argmax confidence 가 아니라 fall_confidence 를 우선 사용한다.
        on_fall_detected(
            confidence=result.get("fall_confidence", result.get("confidence", 0.0)),
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

    rpi_connection = RPiConnection(
        on_status_change=update_rpi4_status,
        on_rescue_request=send_fall_sms,
        on_rpi_log=emit_rpi_log,
    )
    load_dotenv()  # .env에서 환경 변수 로드
    _conf_diag_init()  # [임시 진단] SAFESIGNAL_CONF_DIAG=1이면 매 윈도우 추론값 CSV 기록
    fall_cooldown = FallCooldown(cooldown_sec=int(os.getenv("COOLDOWN_SEC", "30")))
    packet_monitor = PacketMonitor()
    pairing_buffer = PairingBuffer(on_paired=on_paired)

    # 수집 서버 모드: SAFESIGNAL_DISABLE_INFERENCE=1이면 추론 프로세스 자체를
    # 생성/start하지 않는다 (RPCA/추론 부하가 UDP 수신/페어링/대시보드에 주는
    # 영향과 input_queue full/drop 로그 제거). 기본값은 추론 활성.
    inference_disabled = _is_inference_disabled()
    inference_worker = None if inference_disabled else InferenceWorker()
    collect_manager = CollectManager()

    log_info("서버 시작")
    log_info(f"로그 파일: {get_log_filepath()}")
    if inference_disabled:
        log_info("[Inference] disabled by SAFESIGNAL_DISABLE_INFERENCE=1")
    else:
        log_info("[Inference] enabled")

    rpi_connection.start()
    log_info("WebSocket 서버 시작 완료")

    if inference_worker is not None:
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
    # 추론 비활성 시 result_loop는 처리할 결과가 없으므로 스레드를 시작하지 않는다.
    if inference_worker is not None:
        threading.Thread(target=result_loop, daemon=True).start()

    log_info("대시보드: http://localhost:8080")
    start_dashboard(on_fall_detected=on_fall_detected, collect_manager=collect_manager)


# -----------------------------------------------
# 메인 실행
# -----------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main() or 0)
