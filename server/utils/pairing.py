# server/utils/pairing.py

import threading
from typing import Callable, List, Optional, Tuple

from config.settings import BUFFER_MAX_SIZE, DEVICE_ID_RX1, DEVICE_ID_RX2

# 페어링 허용 오차 / 만료 (마이크로초)
# D-007 seq_num은 Rx 장치별 로컬 시퀀스라 한쪽 손실 시 어긋난다 → timestamp 기반 nearest match.
PAIR_TOLERANCE_US = 25_000     # 25ms
PAIR_EXPIRE_US = 200_000       # 200ms


class PairingBuffer:
    """Rx1/Rx2 timestamp_us 기반 nearest-match 페어링.

    동작:
      - add(packet): 반대 Rx 버퍼에서 timestamp_us 차이가 가장 작은 후보를 찾아
        |diff| <= PAIR_TOLERANCE_US 이면 pair 확정.
      - cleanup_expired(): 최근 본 packet timestamp 기준 PAIR_EXPIRE_US 초과 패킷 제거.
      - on_paired 콜백은 항상 Lock 외부에서 호출 → inference put() 등과 데드락 방지.
    """

    def __init__(self, on_paired: Callable[[dict, dict], None]):
        self.on_paired = on_paired
        self._buf_rx1: List[dict] = []
        self._buf_rx2: List[dict] = []
        self._latest_ts_us: int = 0
        self.lock = threading.Lock()

    def add(self, packet: dict) -> None:
        device_id = packet.get("device_id")
        if device_id not in (DEVICE_ID_RX1, DEVICE_ID_RX2):
            return

        ts_us = int(packet.get("timestamp_us", 0))
        paired: Optional[Tuple[dict, dict]] = None

        with self.lock:
            if ts_us > self._latest_ts_us:
                self._latest_ts_us = ts_us

            if device_id == DEVICE_ID_RX1:
                other = self._buf_rx2
                own = self._buf_rx1
            else:
                other = self._buf_rx1
                own = self._buf_rx2

            # 반대 Rx 버퍼에서 가장 가까운 timestamp 탐색
            best_idx = -1
            best_diff = PAIR_TOLERANCE_US + 1
            for i, cand in enumerate(other):
                diff = abs(int(cand["timestamp_us"]) - ts_us)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            if best_idx >= 0 and best_diff <= PAIR_TOLERANCE_US:
                match = other.pop(best_idx)
                if device_id == DEVICE_ID_RX1:
                    paired = (packet, match)
                else:
                    paired = (match, packet)
            else:
                own.append(packet)
                if len(own) > BUFFER_MAX_SIZE:
                    # timestamp가 가장 오래된 패킷부터 제거 (FIFO append이므로 head가 oldest)
                    overflow = len(own) - BUFFER_MAX_SIZE
                    del own[:overflow]

        if paired is not None:
            # Lock 외부에서 콜백 호출 (Queue.put 등과 데드락 방지)
            self.on_paired(paired[0], paired[1])

    def cleanup_expired(self) -> None:
        """가장 최근 본 패킷 timestamp 기준 PAIR_EXPIRE_US 초과 패킷 제거.

        host wall-clock 대신 _latest_ts_us 사용 → ESP↔host SNTP skew 영향 제거.
        """
        with self.lock:
            now_us = self._latest_ts_us
            if now_us == 0:
                return
            self._buf_rx1 = [
                p for p in self._buf_rx1
                if now_us - int(p["timestamp_us"]) <= PAIR_EXPIRE_US
            ]
            self._buf_rx2 = [
                p for p in self._buf_rx2
                if now_us - int(p["timestamp_us"]) <= PAIR_EXPIRE_US
            ]
