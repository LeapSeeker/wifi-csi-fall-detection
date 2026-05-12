# server/collect_manager.py
"""서버 통합 데이터 수집 매니저.

server/main.py의 on_paired() 콜백에서 페어링된 데이터를 받아
SessionRecorder로 CSV 저장하는 래퍼.

사용 흐름:
    1. 대시보드 /collect/start 호출 → start_session()
    2. on_paired() 콜백 → add_pair() (자동)
    3. 대시보드 /collect/stop 호출 → stop_session() → CSV 저장
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import threading
from pathlib import Path
from typing import Optional

from collect.recorder import SessionRecorder
from collect.udp import CsiPacket


class CollectManager:
    """대시보드에서 수집 세션을 제어하는 매니저."""

    def __init__(self, raw_dir: Path = Path("data/raw")):
        self._recorder = SessionRecorder(raw_dir=raw_dir)
        self._is_recording = False
        self._activity_code: Optional[str] = None
        self._env: Optional[int] = None
        self._subject: Optional[int] = None
        self._pair_count = 0
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def pair_count(self) -> int:
        return self._pair_count

    @property
    def activity_code(self) -> Optional[str]:
        return self._activity_code

    def start_session(self, activity_code: str, env: int, subject: int) -> dict:
        """수집 세션 시작."""
        with self._lock:
            if self._is_recording:
                return {"ok": False, "error": "이미 수집 중입니다."}
            self._activity_code = activity_code
            self._env = env
            self._subject = subject
            self._pair_count = 0
            self._is_recording = True
            self._recorder.start_session(activity_code, env, subject)
        return {"ok": True, "activity_code": activity_code, "env": env, "subject": subject}

    def add_pair(self, rx1: dict, rx2: dict) -> None:
        """페어링된 패킷 추가 (on_paired 콜백에서 호출)."""
        with self._lock:
            if not self._is_recording:
                return

        # CsiPacket 형태로 변환
        rx1_pkt = CsiPacket(
            device_id=rx1.get("device_id", 0x01),
            rssi=rx1.get("rssi", 0),
            seq=rx1.get("seq_num", 0),
            timestamp_us=rx1.get("timestamp_us", 0),
            amplitudes=rx1.get("amplitudes", []),
        )
        rx2_pkt = CsiPacket(
            device_id=rx2.get("device_id", 0x02),
            rssi=rx2.get("rssi", 0),
            seq=rx2.get("seq_num", 0),
            timestamp_us=rx2.get("timestamp_us", 0),
            amplitudes=rx2.get("amplitudes", []),
        )
        self._recorder.add_pair(rx1_pkt, rx2_pkt)

        with self._lock:
            self._pair_count += 1

    def stop_session(self, save: bool = True) -> dict:
        """수집 세션 중지 및 저장."""
        with self._lock:
            if not self._is_recording:
                return {"ok": False, "error": "수집 중이 아닙니다."}
            self._is_recording = False
            activity_code = self._activity_code
            env = self._env
            subject = self._subject
            pair_count = self._pair_count

        buf = self._recorder.stop_session()
        loss = self._recorder.calculate_loss_rate(buf)

        if not save or not buf:
            return {
                "ok": True,
                "saved": False,
                "pair_count": pair_count,
                "loss_rate": round(loss * 100, 1),
                "message": "폐기됨" if not save else "수집된 데이터 없음",
            }

        path = self._recorder.save_session(buf, activity_code, env, subject)
        return {
            "ok": True,
            "saved": True,
            "path": str(path) if path else None,
            "pair_count": pair_count,
            "loss_rate": round(loss * 100, 1),
            "message": f"저장됨: {path.name if path else '실패'}",
        }

    def get_status(self) -> dict:
        """현재 수집 상태 반환."""
        with self._lock:
            return {
                "is_recording": self._is_recording,
                "activity_code": self._activity_code,
                "env": self._env,
                "subject": self._subject,
                "pair_count": self._pair_count,
            }

    def get_session_counts(self, env: int, subject: int) -> dict:
        """활동별 수집 완료 횟수 반환."""
        from collect.labels import ACTIVITY_INFO, ACTIVITY_ORDER
        counts = {}
        for code in ACTIVITY_ORDER:
            counts[code] = {
                "done": self._recorder.count_sessions(code, env, subject),
                "target": ACTIVITY_INFO[code]["target"],
                "display": ACTIVITY_INFO[code]["display"],
            }
        return counts