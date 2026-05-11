"""server/inference 모듈 self-check.

torch / model checkpoint / RPCA 없이 동작해야 한다. 즉:
- worker.py top-level 에서 predictor 를 import 하지 않는다.
- InferenceWorker self-check 는 start() 를 호출하지 않는다.

실행:
    python server/inference/_selfcheck.py
마지막 출력:
    ALL_OK
"""
from __future__ import annotations

import socket
import struct
import sys
from pathlib import Path

# project_root/server 를 path 에 추가 (config.* 등 server 모듈 import 용)
_HERE = Path(__file__).resolve()
_INFERENCE_DIR = _HERE.parent
_SERVER_DIR = _HERE.parents[1]
_PROJECT_ROOT = _HERE.parents[2]

# 스크립트 dir 인 server/inference 가 sys.path 최상단에 있으면
# server/config/ 네임스페이스 패키지보다 server/inference/config.py 가
# 먼저 매치돼 `from config.settings import ...` 가 깨진다. 제거.
_script_dir = str(_INFERENCE_DIR)
while _script_dir in sys.path:
    sys.path.remove(_script_dir)

for p in (str(_SERVER_DIR), str(_PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from receiver.udp_receiver import (
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC_BYTE,
    PACKET_SIZE,
    parse_packet,
    start_receivers,
)
from config.settings import DEVICE_ID_RX1, DEVICE_ID_RX2, SUBCARRIER_COUNT
from utils.pairing import (
    PAIR_EXPIRE_US,
    PAIR_TOLERANCE_US,
    PairingBuffer,
)

from inference.buffer import SlidingWindowBuffer  # noqa: E402
from inference.config import (
    INFERENCE_STRIDE,
    MODEL_PATH,
    N_SUBCARRIERS_EACH,
    WINDOW_SIZE,
)
from inference.worker import InferenceWorker  # noqa: E402


def _build_packet(
    device_id: int = DEVICE_ID_RX1,
    rssi: int = -50,
    seq: int = 1234,
    timestamp_us: int = 1_700_000_000_000_000,
    n_sc: int = SUBCARRIER_COUNT,
    magic: int = MAGIC_BYTE,
) -> bytes:
    header = struct.pack(HEADER_FORMAT, magic, device_id, rssi, 0, seq, timestamp_us)
    amps = struct.pack(f"<{n_sc}f", *[0.5] * n_sc)
    return header + amps


# -----------------------------------------------------------------
# 1) D-007 packet parsing
# -----------------------------------------------------------------
def check_packet_parsing() -> bool:
    # 정상 패킷
    raw = _build_packet(device_id=DEVICE_ID_RX1, rssi=-42, seq=100,
                        timestamp_us=1234567890)
    assert len(raw) == PACKET_SIZE, f"size {len(raw)} != {PACKET_SIZE}"
    pkt = parse_packet(raw)
    assert pkt is not None, "정상 패킷이 None 반환"
    assert pkt["device_id"] == DEVICE_ID_RX1
    assert pkt["rssi"] == -42
    assert pkt["seq_num"] == 100
    assert pkt["timestamp_us"] == 1234567890
    assert pkt["n_subcarriers"] == SUBCARRIER_COUNT
    assert len(pkt["amplitudes"]) == SUBCARRIER_COUNT
    assert all(abs(a - 0.5) < 1e-6 for a in pkt["amplitudes"])

    # Rx2 정상 패킷
    raw_rx2 = _build_packet(device_id=DEVICE_ID_RX2)
    assert parse_packet(raw_rx2) is not None

    # 크기 부족 (223B)
    assert parse_packet(raw[:-1]) is None, "223B None 미반환"
    assert parse_packet(b"") is None
    assert parse_packet(b"\x00" * (HEADER_SIZE - 1)) is None

    # 크기 초과 (225B) — strict 224B check
    raw_too_big = raw + b"\x00"
    assert len(raw_too_big) == PACKET_SIZE + 1
    assert parse_packet(raw_too_big) is None, "225B None 미반환 (strict 224B 위반)"

    # magic 불일치
    bad_magic = _build_packet(magic=0x00)
    assert parse_packet(bad_magic) is None, "magic 불일치 None 미반환"

    # device_id 불일치 — 0x99 (정의되지 않은 값)
    bad_dev = _build_packet(device_id=0x99)
    assert parse_packet(bad_dev) is None, "잘못된 device_id None 미반환"

    # device_id = 0x00 (Tx) 도 거부되어야 한다
    tx_dev = _build_packet(device_id=0x00)
    assert parse_packet(tx_dev) is None, "Tx(0x00) device_id None 미반환"

    print("  [1] packet parsing OK (strict 224B)")
    return True


# -----------------------------------------------------------------
# 2) SlidingWindowBuffer shape & stride trigger
# -----------------------------------------------------------------
def check_buffer() -> bool:
    buf = SlidingWindowBuffer()
    assert buf.n_cols == N_SUBCARRIERS_EACH * 2

    rx1 = [1.0] * N_SUBCARRIERS_EACH
    rx2 = [2.0] * N_SUBCARRIERS_EACH

    trigger_indices = []
    for i in range(1, WINDOW_SIZE + INFERENCE_STRIDE * 2 + 1):
        triggered = buf.add(rx1, rx2)
        if triggered:
            trigger_indices.append(i)

    # 첫 trigger는 WINDOW_SIZE 도달 시점 (300번째 add)
    # 이후 INFERENCE_STRIDE(100) 간격으로 발생: 300, 400, 500
    expected = [WINDOW_SIZE,
                WINDOW_SIZE + INFERENCE_STRIDE,
                WINDOW_SIZE + INFERENCE_STRIDE * 2]
    assert trigger_indices == expected, (
        f"trigger indices {trigger_indices} != expected {expected}"
    )

    window = buf.get_window()
    assert window.shape == (WINDOW_SIZE, N_SUBCARRIERS_EACH * 2), \
        f"shape {window.shape}"
    assert window.dtype == np.float32, f"dtype {window.dtype}"

    # 잘못된 길이 입력은 False
    assert buf.add([1.0] * 10, rx2) is False
    assert buf.add(rx1, [1.0] * 10) is False

    print("  [2] SlidingWindowBuffer shape & stride OK")
    return True


# -----------------------------------------------------------------
# 3) MODEL_PATH resolution
# -----------------------------------------------------------------
def check_model_path() -> bool:
    assert isinstance(MODEL_PATH, Path), f"MODEL_PATH type {type(MODEL_PATH)}"
    assert MODEL_PATH.is_absolute(), f"MODEL_PATH not absolute: {MODEL_PATH}"
    assert MODEL_PATH.name == "best.pt", f"filename {MODEL_PATH.name}"
    expected_suffix = Path("model") / "pretrained" / "checkpoints" / "best.pt"
    assert str(MODEL_PATH).endswith(str(expected_suffix)), (
        f"MODEL_PATH={MODEL_PATH} does not end with {expected_suffix}"
    )
    # 파일 존재 여부는 확인 안 함 (학습 후 생성)
    print(f"  [3] MODEL_PATH = {MODEL_PATH}  (existence not required)")
    return True


# -----------------------------------------------------------------
# 4) InferenceWorker queue non-blocking (start() 없이)
# -----------------------------------------------------------------
def check_worker_queue() -> bool:
    worker = InferenceWorker()
    assert worker._process is None, "init 시 process is not None"

    dummy_rx1 = {
        "amplitudes": [0.1] * N_SUBCARRIERS_EACH,
        "seq_num": 42,
        "timestamp_us": 1000,
    }
    dummy_rx2 = {
        "amplitudes": [0.2] * N_SUBCARRIERS_EACH,
        "seq_num": 42,
        "timestamp_us": 1000,
    }

    # put() 예외 없이 호출
    worker.put(dummy_rx1, dummy_rx2)

    # get_result()는 비어 있어 None
    result = worker.get_result()
    assert result is None, f"empty queue가 {result} 반환"

    # 큐 리소스 정리
    try:
        worker.input_queue.close()
        worker.input_queue.join_thread()
        worker.result_queue.close()
        worker.result_queue.join_thread()
    except Exception:
        pass

    print("  [4] InferenceWorker put/get_result non-blocking OK")
    return True


# -----------------------------------------------------------------
# 5) PairingBuffer timestamp 기반 페어링
# -----------------------------------------------------------------
def _mk_pkt(device_id: int, ts_us: int, seq: int = 0, rssi: int = -50) -> dict:
    return {
        "device_id": device_id,
        "rssi": rssi,
        "seq_num": seq,
        "timestamp_us": ts_us,
        "n_subcarriers": SUBCARRIER_COUNT,
        "amplitudes": [0.5] * SUBCARRIER_COUNT,
    }


def check_pairing_buffer() -> bool:
    # 5-1) Rx1 → Rx2 (30ms 차이) → pair 성공
    pairs: list = []
    buf = PairingBuffer(on_paired=lambda r1, r2: pairs.append((r1, r2)))
    buf.add(_mk_pkt(DEVICE_ID_RX1, ts_us=1_000_000))
    buf.add(_mk_pkt(DEVICE_ID_RX2, ts_us=1_030_000))
    assert len(pairs) == 1, f"30ms 차이 pair 실패: {len(pairs)}"
    rx1, rx2 = pairs[0]
    assert rx1["device_id"] == DEVICE_ID_RX1, "콜백 첫 인자 != Rx1"
    assert rx2["device_id"] == DEVICE_ID_RX2, "콜백 둘째 인자 != Rx2"

    # 5-2) tolerance 초과 (60.001ms) → pair 실패
    pairs.clear()
    buf2 = PairingBuffer(on_paired=lambda r1, r2: pairs.append((r1, r2)))
    buf2.add(_mk_pkt(DEVICE_ID_RX1, ts_us=2_000_000))
    buf2.add(_mk_pkt(DEVICE_ID_RX2, ts_us=2_060_001))
    assert len(pairs) == 0, f"50ms 초과인데 pair 됨: {len(pairs)}"
    # 두 버퍼 모두 잔여 1개
    assert len(buf2._buf_rx1) == 1
    assert len(buf2._buf_rx2) == 1

    # 5-3) Rx2 먼저 → Rx1 나중 (20ms 차) → pair 성공, 순서 (rx1, rx2)
    pairs.clear()
    buf3 = PairingBuffer(on_paired=lambda r1, r2: pairs.append((r1, r2)))
    buf3.add(_mk_pkt(DEVICE_ID_RX2, ts_us=3_000_000, seq=10))
    buf3.add(_mk_pkt(DEVICE_ID_RX1, ts_us=3_020_000, seq=20))
    assert len(pairs) == 1, "역순 pair 실패"
    rx1, rx2 = pairs[0]
    assert rx1["device_id"] == DEVICE_ID_RX1, "역순 콜백 첫 인자 != Rx1"
    assert rx2["device_id"] == DEVICE_ID_RX2, "역순 콜백 둘째 인자 != Rx2"
    assert rx1["seq_num"] == 20 and rx2["seq_num"] == 10, "역순 packet body 어긋남"

    # 5-4) cleanup_expired — _latest_ts_us 기준 PAIR_EXPIRE_US 초과 제거
    buf4 = PairingBuffer(on_paired=lambda r1, r2: None)
    buf4.add(_mk_pkt(DEVICE_ID_RX1, ts_us=10_000_000))      # 오래된 packet
    buf4.add(_mk_pkt(DEVICE_ID_RX1, ts_us=10_000_000 + PAIR_EXPIRE_US + 1))  # 최신
    # 두 packet 모두 Rx1이므로 pair 안 됨 → 둘 다 _buf_rx1 잔존
    assert len(buf4._buf_rx1) == 2
    buf4.cleanup_expired()
    # 오래된 한 개만 제거되어야 함
    assert len(buf4._buf_rx1) == 1, (
        f"cleanup 후 잔여 {len(buf4._buf_rx1)} (기대 1)"
    )

    # cleanup 빈 상태 — _latest_ts_us == 0이면 즉시 반환
    buf_empty = PairingBuffer(on_paired=lambda r1, r2: None)
    buf_empty.cleanup_expired()  # 예외 없음

    # 5-5) 버퍼 cap 초과 → 오래된 패킷부터 제거
    from config.settings import BUFFER_MAX_SIZE
    buf5 = PairingBuffer(on_paired=lambda r1, r2: None)
    # tolerance 멀리 떨어진 Rx1 패킷을 BUFFER_MAX_SIZE+5개 add → pair 없이 누적
    base = 100_000_000
    for i in range(BUFFER_MAX_SIZE + 5):
        # 각 패킷 사이 PAIR_TOLERANCE_US*4 간격 → 절대 self-pair X
        buf5.add(_mk_pkt(DEVICE_ID_RX1, ts_us=base + i * PAIR_TOLERANCE_US * 4, seq=i))
    assert len(buf5._buf_rx1) == BUFFER_MAX_SIZE, (
        f"cap 초과 후 길이 {len(buf5._buf_rx1)} != {BUFFER_MAX_SIZE}"
    )
    # 가장 오래된 패킷(seq=0..4) 제거되고 최신 5개 유지
    seq_remaining = [p["seq_num"] for p in buf5._buf_rx1]
    assert seq_remaining[0] == 5, f"oldest packet 미제거: head seq={seq_remaining[0]}"
    assert seq_remaining[-1] == BUFFER_MAX_SIZE + 4

    # 5-6) device_id 잘못된 packet은 무시
    buf6 = PairingBuffer(on_paired=lambda r1, r2: pairs.append((r1, r2)))
    pairs.clear()
    buf6.add(_mk_pkt(0x99, ts_us=1))
    buf6.add(_mk_pkt(0x00, ts_us=2))
    assert len(buf6._buf_rx1) == 0 and len(buf6._buf_rx2) == 0
    assert len(pairs) == 0

    print("  [5] PairingBuffer timestamp 기반 페어링 OK")
    return True


# -----------------------------------------------------------------
# 6) UDP bind 실패 시 RuntimeError
# -----------------------------------------------------------------
def check_udp_bind_failure() -> bool:
    # ephemeral port 할당 (("0.0.0.0", 0)) → 그 포트 점유 후 start_receivers 호출
    blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    blocker.bind(("0.0.0.0", 0))
    occupied_port = blocker.getsockname()[1]

    try:
        try:
            start_receivers(callback=lambda p: None, port=occupied_port)
        except RuntimeError as e:
            msg = str(e)
            assert str(occupied_port) in msg, (
                f"에러 메시지에 포트 번호 없음: {msg}"
            )
            assert "바인드" in msg or "bind" in msg.lower(), (
                f"에러 메시지에 bind 표현 없음: {msg}"
            )
            # 원본 예외 chain 확인
            assert isinstance(e.__cause__, OSError), (
                f"raise ... from OSError 미적용: __cause__={type(e.__cause__)}"
            )
            print(f"  [6] UDP bind RuntimeError OK (port={occupied_port})")
            return True
        else:
            raise AssertionError("점유된 포트에 start_receivers가 RuntimeError 미발생")
    finally:
        blocker.close()


def main() -> int:
    print("== server self-check ==")
    check_packet_parsing()
    check_buffer()
    check_model_path()
    check_worker_queue()
    check_pairing_buffer()
    check_udp_bind_failure()
    print("ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
