"""SafeSignal 자체 데이터 수집 — 비프음 (Windows 표준 라이브러리)."""

from __future__ import annotations

import sys


def _beep(freq: int, ms: int) -> None:
    if sys.platform == "win32":
        import winsound

        winsound.Beep(freq, ms)
    else:
        # 비-Windows 환경 fallback (기능 자체는 Windows 전용)
        print(f"[BEEP {freq}Hz {ms}ms]", flush=True)


def beep_ready() -> None:
    """준비 비프 — 500Hz, 200ms."""
    _beep(500, 200)


def beep_stage() -> None:
    """단계 전환 비프 — 1000Hz, 500ms."""
    _beep(1000, 500)


def beep_end() -> None:
    """종료 비프 — 1500Hz, 1000ms."""
    _beep(1500, 1000)
