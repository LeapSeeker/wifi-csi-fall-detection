"""SafeSignal 자체 데이터 수집 — UDP 수신 + Rx1/Rx2 페어링.

펌웨어 (firmware/csi_rx1, csi_rx2)와 STATE.md D-007 기준:
- 패킷 224B = magic(0xAB) | device_id | rssi(int8) | reserved | seq(u32)
                | timestamp_us(u64) | amplitude×52 (float32, 208B)
- 포트 5005, packed (no padding)
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

# ── 패킷 포맷 상수 ────────────────────────────────────────
HEADER_FORMAT = "<BBbBIQ"  # magic, device_id, rssi(int8), reserved, seq, ts_us
DEVICE_ID_RX1 = 0x01
DEVICE_ID_RX2 = 0x02
PACKET_MAGIC = 0xAB
UDP_PORT = 5005
PACKET_SIZE = 224
HEADER_SIZE = 16  # struct.calcsize(HEADER_FORMAT)
AMPLITUDE_COUNT = 52
AMPLITUDE_SIZE = AMPLITUDE_COUNT * 4  # 208

assert struct.calcsize(HEADER_FORMAT) == HEADER_SIZE, "HEADER_FORMAT size mismatch"
assert HEADER_SIZE + AMPLITUDE_SIZE == PACKET_SIZE, "PACKET_SIZE mismatch"

# ── 페어링 파라미터 ────────────────────────────────────────
PAIR_TOLERANCE_US = 25_000     # 25ms
EXPIRE_US = 200_000            # 200ms 만료
BUFFER_MAX = 200               # 한쪽 버퍼 최대 크기
CLEANUP_INTERVAL_S = 0.05      # 50ms 주기 cleanup


@dataclass
class CsiPacket:
    device_id: int
    rssi: int
    seq: int
    timestamp_us: int
    amplitudes: list[float]


def parse_packet(data: bytes) -> Optional[CsiPacket]:
    """UDP raw bytes → CsiPacket. 유효하지 않으면 None."""
    if len(data) != PACKET_SIZE:
        return None

    try:
        magic, device_id, rssi, _reserved, seq, ts_us = struct.unpack_from(
            HEADER_FORMAT, data, 0
        )
    except struct.error:
        return None

    if magic != PACKET_MAGIC:
        return None
    if device_id not in (DEVICE_ID_RX1, DEVICE_ID_RX2):
        return None

    amplitudes = list(
        struct.unpack_from(f"<{AMPLITUDE_COUNT}f", data, HEADER_SIZE)
    )
    if len(amplitudes) != AMPLITUDE_COUNT:
        return None

    return CsiPacket(
        device_id=device_id,
        rssi=rssi,
        seq=seq,
        timestamp_us=ts_us,
        amplitudes=amplitudes,
    )


class PairingBuffer:
    """Rx1/Rx2 timestamp 기반 페어링 버퍼.

    수신된 한쪽 패킷에 대해 반대 Rx 버퍼에서 timestamp가 가장 가까운 패킷을
    탐색하고, 차이가 PAIR_TOLERANCE_US 이내일 때만 pair로 확정한다.
    """

    def __init__(self, on_paired: Callable[[CsiPacket, CsiPacket], None]):
        self._on_paired = on_paired
        self._buf_rx1: list[CsiPacket] = []
        self._buf_rx2: list[CsiPacket] = []
        self._latest_ts_us = 0
        self._lock = threading.Lock()

    def _other(self, device_id: int) -> list[CsiPacket]:
        return self._buf_rx2 if device_id == DEVICE_ID_RX1 else self._buf_rx1

    def _own(self, device_id: int) -> list[CsiPacket]:
        return self._buf_rx1 if device_id == DEVICE_ID_RX1 else self._buf_rx2

    def add(self, pkt: CsiPacket) -> None:
        paired: Optional[tuple[CsiPacket, CsiPacket]] = None
        with self._lock:
            if pkt.timestamp_us > self._latest_ts_us:
                self._latest_ts_us = pkt.timestamp_us
            other = self._other(pkt.device_id)
            own = self._own(pkt.device_id)

            # 반대 Rx 버퍼에서 가장 가까운 timestamp 탐색
            best_idx = -1
            best_diff = PAIR_TOLERANCE_US + 1
            for i, cand in enumerate(other):
                diff = abs(cand.timestamp_us - pkt.timestamp_us)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            if best_idx >= 0 and best_diff <= PAIR_TOLERANCE_US:
                match = other.pop(best_idx)
                if pkt.device_id == DEVICE_ID_RX1:
                    paired = (pkt, match)
                else:
                    paired = (match, pkt)
            else:
                own.append(pkt)
                if len(own) > BUFFER_MAX:
                    # 가장 오래된 패킷부터 제거
                    del own[: len(own) - BUFFER_MAX]

        if paired is not None:
            self._on_paired(paired[0], paired[1])

    def cleanup(self) -> None:
        """가장 최근 패킷 timestamp 기준 EXPIRE_US 초과 미완성 패킷 제거."""
        with self._lock:
            now_us = self._latest_ts_us
            if now_us == 0:
                return
            self._buf_rx1 = [
                p for p in self._buf_rx1 if now_us - p.timestamp_us <= EXPIRE_US
            ]
            self._buf_rx2 = [
                p for p in self._buf_rx2 if now_us - p.timestamp_us <= EXPIRE_US
            ]


def start_udp_receiver(
    on_paired: Callable[[CsiPacket, CsiPacket], None],
    port: int = UDP_PORT,
) -> tuple[threading.Thread, threading.Thread]:
    """UDP 수신 스레드 + cleanup 스레드를 daemon으로 실행한다.

    server/dongseok과 같은 UDP 포트를 동시에 bind하지 않는다(포트 공유 시도 X).
    """
    pair_buf = PairingBuffer(on_paired)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as e:
        sock.close()
        raise RuntimeError(
            f"UDP 포트 {port}에 바인드할 수 없습니다. "
            "server/dongseok 수신 서버 또는 이전 collect 프로세스가 "
            "이미 실행 중인지 확인하세요. 다른 포트를 쓰려면 --port 옵션을 사용하세요."
        ) from e

    def _recv_loop() -> None:
        while True:
            try:
                data, _addr = sock.recvfrom(4096)
            except OSError:
                continue
            pkt = parse_packet(data)
            if pkt is None:
                continue
            pair_buf.add(pkt)

    def _cleanup_loop() -> None:
        while True:
            time.sleep(CLEANUP_INTERVAL_S)
            pair_buf.cleanup()

    t_recv = threading.Thread(target=_recv_loop, name="csi-udp-recv", daemon=True)
    t_clean = threading.Thread(
        target=_cleanup_loop, name="csi-udp-cleanup", daemon=True
    )
    t_recv.start()
    t_clean.start()
    return t_recv, t_clean
