"""SafeSignal 자체 데이터 수집 — SessionRecorder.

페어링된 (rx1, rx2)를 한 행으로 누적하고, 세션 종료 시 CSV로 저장한다.
파일명 규칙: E{env}_S{subj:02d}_A_{activity_code}_T{trial:03d}.csv
저장 경로: data/raw/

CSV 컬럼은 110개로 고정 (rssi는 패킷 파싱만 유지하고 CSV/전처리 입력에서는 제외).
timestamp_us는 호환성을 위해 Rx1 timestamp 의미를 그대로 유지하며,
timestamp_rx1_us / timestamp_rx2_us / pair_dt_us는 페어링 품질 사후 검증용으로 추가됐다.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from collect.udp import AMPLITUDE_COUNT, CsiPacket

DATA_RAW_DIR = Path("data/raw")

# CSV 컬럼 순서 (총 110개) — rssi는 패킷에서 파싱하지만 저장하지 않음.
# 페어링 품질 컬럼(timestamp_rx1_us/timestamp_rx2_us/pair_dt_us)은 맨 끝에 append.
_AMP_RX1_COLS = [f"amp_rx1_{i}" for i in range(AMPLITUDE_COUNT)]
_AMP_RX2_COLS = [f"amp_rx2_{i}" for i in range(AMPLITUDE_COUNT)]
CSV_COLUMNS: list[str] = (
    ["timestamp_us", "seq_rx1", "seq_rx2"]
    + _AMP_RX1_COLS
    + _AMP_RX2_COLS
    + ["timestamp_rx1_us", "timestamp_rx2_us", "pair_dt_us"]
)


class SessionRecorder:
    """단일 세션 단위로 페어링 패킷을 누적/저장하는 레코더."""

    def __init__(self, raw_dir: Path = DATA_RAW_DIR):
        self._raw_dir = raw_dir
        self._buf: list[dict] = []
        self._is_recording = False
        self._activity_code: Optional[str] = None
        self._env: Optional[int] = None
        self._subject: Optional[int] = None
        self._lock = threading.Lock()

    # ── 세션 제어 ────────────────────────────────────────
    def start_session(self, activity_code: str, env: int, subject: int) -> None:
        with self._lock:
            self._buf.clear()
            self._activity_code = activity_code
            self._env = env
            self._subject = subject
            self._is_recording = True

    def add_pair(self, rx1: CsiPacket, rx2: CsiPacket) -> None:
        with self._lock:
            if not self._is_recording:
                return
            row = {
                "timestamp_us": rx1.timestamp_us,  # 호환성: Rx1 timestamp 의미 유지
                "seq_rx1": rx1.seq,
                "seq_rx2": rx2.seq,
            }
            for i, v in enumerate(rx1.amplitudes):
                row[f"amp_rx1_{i}"] = v
            for i, v in enumerate(rx2.amplitudes):
                row[f"amp_rx2_{i}"] = v
            # 페어링 품질 사후 검증용 (report-only)
            row["timestamp_rx1_us"] = rx1.timestamp_us
            row["timestamp_rx2_us"] = rx2.timestamp_us
            row["pair_dt_us"] = abs(rx1.timestamp_us - rx2.timestamp_us)
            self._buf.append(row)

    def stop_session(self) -> list[dict]:
        with self._lock:
            self._is_recording = False
            buf_copy = list(self._buf)
            self._buf.clear()
            return buf_copy

    # ── 분석 헬퍼 ────────────────────────────────────────
    @staticmethod
    def calculate_loss_rate(buf: list[dict]) -> float:
        """rx1/rx2 seq 기준 손실률 중 더 높은 값을 반환 (0~1)."""
        if len(buf) <= 1:
            return 0.0

        def _loss(seqs: list[int]) -> float:
            if not seqs:
                return 0.0
            lo, hi = min(seqs), max(seqs)
            expected = hi - lo + 1
            if expected <= 0:
                return 0.0
            unique = len(set(seqs))
            return (expected - unique) / expected

        loss_rx1 = _loss([row["seq_rx1"] for row in buf])
        loss_rx2 = _loss([row["seq_rx2"] for row in buf])
        return max(loss_rx1, loss_rx2)

    # ── 저장 / 카운트 ────────────────────────────────────
    def _make_filename(self, activity_code: str, env: int, subject: int, trial: int) -> Path:
        return (
            self._raw_dir
            / f"E{env}_S{subject:02d}_A_{activity_code}_T{trial:03d}.csv"
        )

    def _existing_trials(self, activity_code: str, env: int, subject: int) -> list[int]:
        if not self._raw_dir.exists():
            return []
        pattern = re.compile(
            rf"^E{env}_S{subject:02d}_A_{re.escape(activity_code)}_T(\d+)\.csv$"
        )
        trials: list[int] = []
        for p in self._raw_dir.glob(
            f"E{env}_S{subject:02d}_A_{activity_code}_T*.csv"
        ):
            m = pattern.match(p.name)
            if m:
                trials.append(int(m.group(1)))
        return trials

    def _next_trial(self, activity_code: str, env: int, subject: int) -> int:
        trials = self._existing_trials(activity_code, env, subject)
        return max(trials) + 1 if trials else 1

    def save_session(
        self,
        buf: list[dict],
        activity_code: str,
        env: int,
        subject: int,
    ) -> Optional[Path]:
        if not buf:
            print("Session empty, file not saved.")
            return None

        self._raw_dir.mkdir(parents=True, exist_ok=True)
        trial = self._next_trial(activity_code, env, subject)
        path = self._make_filename(activity_code, env, subject, trial)
        # 동일 경로 존재 시 1씩 증가하며 빈 번호 탐색 (race / 비정상 잔존 파일 방지)
        while path.exists():
            trial += 1
            path = self._make_filename(activity_code, env, subject, trial)
        df = pd.DataFrame(buf, columns=CSV_COLUMNS)
        df.to_csv(path, index=False)
        return path

    def count_sessions(self, activity_code: str, env: int, subject: int) -> int:
        if not self._raw_dir.exists():
            return 0
        return len(
            list(
                self._raw_dir.glob(
                    f"E{env}_S{subject:02d}_A_{activity_code}_T*.csv"
                )
            )
        )
