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
)
from config.settings import DEVICE_ID_RX1, DEVICE_ID_RX2, SUBCARRIER_COUNT

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

    # 크기 부족
    assert parse_packet(raw[:-1]) is None, "크기 부족 None 미반환"
    assert parse_packet(b"") is None
    assert parse_packet(b"\x00" * (HEADER_SIZE - 1)) is None

    # magic 불일치
    bad_magic = _build_packet(magic=0x00)
    assert parse_packet(bad_magic) is None, "magic 불일치 None 미반환"

    # device_id 불일치
    bad_dev = _build_packet(device_id=0x99)
    assert parse_packet(bad_dev) is None, "잘못된 device_id None 미반환"

    print("  [1] packet parsing OK")
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


def main() -> int:
    print("== server/inference self-check ==")
    check_packet_parsing()
    check_buffer()
    check_model_path()
    check_worker_queue()
    print("ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
