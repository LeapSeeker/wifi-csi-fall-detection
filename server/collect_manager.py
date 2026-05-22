# server/collect_manager.py
"""서버 통합 데이터 수집 매니저.

server/main.py의 on_paired() 콜백에서 페어링된 데이터를 받아
SessionRecorder로 CSV 저장하는 래퍼.

사용 흐름:
    1. 대시보드 /collect/start 호출 → start_session()
    2. on_paired() 콜백 → add_pair() (자동)
    3. 녹화 자동 종료 후 저장/폐기 버튼 클릭
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional, Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from collect.recorder import SessionRecorder
from collect.labels import ACTIVITY_INFO
from collect.udp import CsiPacket
from collect.drive_upload import upload_file_async


def _beep(freq: int, ms: int) -> None:
    try:
        if sys.platform == "win32":
            import winsound
            winsound.Beep(freq, ms)
        else:
            print(f"[BEEP {freq}Hz {ms}ms]", flush=True)
    except Exception:
        pass


def beep_ready() -> None:
    _beep(500, 200)


def beep_stage() -> None:
    _beep(1000, 500)


def beep_end() -> None:
    _beep(1500, 1000)


class CollectManager:
    """대시보드에서 수집 세션을 제어하는 매니저."""

    def __init__(self, raw_dir: Path | None = None):
        self._recorder = SessionRecorder(raw_dir=raw_dir or (_PROJECT_ROOT / "data" / "raw"))
        self._is_recording = False
        self._activity_code: Optional[str] = None
        self._env: Optional[int] = None
        self._subject: Optional[int] = None
        self._pair_count = 0
        self._pending_buf: list = []
        self._lock = threading.Lock()
        self._emit_fn: Optional[Callable] = None

    def set_emit(self, emit_fn: Callable) -> None:
        """socketio.emit 함수 주입 (app.py에서 호출)."""
        self._emit_fn = emit_fn

    def _emit(self, event: str, data: dict) -> None:
        if self._emit_fn:
            try:
                self._emit_fn(event, data)
            except Exception:
                pass

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
        """수집 세션 시작 — 카운트다운 후 녹화 시작."""
        if activity_code not in ACTIVITY_INFO:
            return {"ok": False, "error": f"알 수 없는 activity_code: {activity_code}"}

        with self._lock:
            if self._is_recording:
                return {"ok": False, "error": "이미 수집 중입니다."}
            self._activity_code = activity_code
            self._env = env
            self._subject = subject
            self._pair_count = 0
            self._pending_buf = []

        threading.Thread(
            target=self._run_session,
            args=(activity_code, env, subject),
            daemon=True,
        ).start()
        return {"ok": True, "activity_code": activity_code, "env": env, "subject": subject}

    def _run_session(self, activity_code: str, env: int, subject: int) -> None:
        """카운트다운 → 녹화 → stage 루프 → 종료."""
        info = ACTIVITY_INFO[activity_code]

        # 1. 준비 비프 + 3초 카운트다운
        beep_ready()
        for i in range(3, 0, -1):
            self._emit("collect_countdown", {"count": i})
            time.sleep(1)
        self._emit("collect_countdown", {"count": 0})

        # 2. 녹화 시작
        with self._lock:
            self._is_recording = True
        self._recorder.start_session(activity_code, env, subject)
        self._emit("collect_recording_started", {
            "activity_code": activity_code,
            "env": env,
            "subject": subject,
        })

        # 3. stage 루프
        if "stages" in info:
            for i, stage in enumerate(info["stages"]):
                beep_stage()
                self._emit("collect_stage", {
                    "stage_idx": i,
                    "stage_name": stage["name"],
                    "duration": stage["duration"],
                    "total_stages": len(info["stages"]),
                })
                time.sleep(stage["duration"])
        else:
            duration = info["duration"]
            beep_stage()
            self._emit("collect_stage", {
                "stage_idx": 0,
                "stage_name": info["display"],
                "duration": duration,
                "total_stages": 1,
            })
            time.sleep(duration)

        # 4. 녹화 자동 종료
        with self._lock:
            self._is_recording = False
        buf = self._recorder.stop_session()
        self._pending_buf = buf
        beep_end()

        loss = self._recorder.calculate_loss_rate(buf)
        self._emit("collect_finished", {
            "pair_count": len(buf),
            "loss_rate": round(loss * 100, 1),
            "warn": loss > 0.05,
        })

    def save_pending(self) -> dict:
        """녹화 완료 후 저장."""
        buf = self._pending_buf
        if not buf:
            return {"ok": False, "error": "저장할 데이터 없음"}
        path = self._recorder.save_session(
            buf, self._activity_code, self._env, self._subject
        )
        if path is not None:
            upload_file_async(path)
        self._pending_buf = []
        loss = self._recorder.calculate_loss_rate(buf)
        return {
            "ok": True, "saved": True,
            "path": str(path) if path else None,
            "pair_count": len(buf),
            "loss_rate": round(loss * 100, 1),
            "message": f"저장됨: {path.name if path else '실패'}",
        }

    def discard_pending(self) -> dict:
        """녹화 완료 후 폐기."""
        count = len(self._pending_buf)
        self._pending_buf = []
        return {"ok": True, "saved": False, "pair_count": count, "message": "폐기됨"}

    def stop_session(self, save: bool = True) -> dict:
        """수동 중지 또는 자동 종료 후 저장/폐기."""
        with self._lock:
            is_recording = self._is_recording

        if not is_recording:
            # 자동 종료 후 저장/폐기
            return self.save_pending() if save else self.discard_pending()

        # 수동 중지
        with self._lock:
            self._is_recording = False
            activity_code = self._activity_code
            env = self._env
            subject = self._subject
            pair_count = self._pair_count

        buf = self._recorder.stop_session()
        beep_end()
        loss = self._recorder.calculate_loss_rate(buf)

        if not save or not buf:
            return {
                "ok": True, "saved": False,
                "pair_count": pair_count,
                "loss_rate": round(loss * 100, 1),
                "message": "폐기됨" if not save else "수집된 데이터 없음",
            }

        path = self._recorder.save_session(buf, activity_code, env, subject)
        if path is not None:
            upload_file_async(path)
        return {
            "ok": True, "saved": True,
            "path": str(path) if path else None,
            "pair_count": pair_count,
            "loss_rate": round(loss * 100, 1),
            "message": f"저장됨: {path.name if path else '실패'}",
        }

    def add_pair(self, rx1: dict, rx2: dict) -> None:
        """페어링된 패킷 추가 (on_paired 콜백에서 호출)."""
        with self._lock:
            if not self._is_recording:
                return

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

    def get_status(self) -> dict:
        with self._lock:
            return {
                "is_recording": self._is_recording,
                "activity_code": self._activity_code,
                "env": self._env,
                "subject": self._subject,
                "pair_count": self._pair_count,
            }

    def get_session_counts(self, env: int, subject: int) -> dict:
        counts = {}
        for code, info in ACTIVITY_INFO.items():
            counts[code] = {
                "done": self._recorder.count_sessions(code, env, subject),
                "target": info["target"],
                "display": info["display"],
            }
        return counts
