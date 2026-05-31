"""SafeSignal 수집 품질 요약 — pair delay / timestamp gap 통계.

CLI(collect_main)와 서버(collect_manager)가 동일 기준으로 사용한다.
입력은 SessionRecorder buffer (list[dict]) 또는 동등한 행 dict 리스트.

report-only: 본 모듈이 계산하는 pair_dt / gap 값은 저장 차단이나 RECOLLECT
판정에 강하게 연결하지 않는다. 실측 분포 확보 후 별도 threshold를 정한다.

legacy 107컬럼 버퍼(= pair_dt_us 없음)와 신규 110컬럼 버퍼를 모두 처리하며,
row 수가 0 또는 1이어도 예외 없이 안전하게 동작한다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from collect.recorder import SessionRecorder

CAPTURE_TARGET_HZ = 100.0


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize_session(buf: list[dict]) -> dict:
    """수집 버퍼의 품질 요약을 계산한다.

    반환 키:
        pair_count, duration_s, pair_rate_hz, capture_ratio, loss_rate,
        pair_dt_p50_us, pair_dt_p95_us, pair_dt_p99_us, pair_dt_max_us,
        gap_p95_us, gap_max_us, ts_reversal_count, ts_min_signed_gap_us

    pair_dt_us / gap을 계산할 수 없으면(legacy 버퍼, row 부족) 해당 값은 None.
    """
    n = len(buf)
    summary: dict = {
        "pair_count": n,
        "duration_s": 0.0,
        "pair_rate_hz": 0.0,
        "capture_ratio": 0.0,
        "loss_rate": SessionRecorder.calculate_loss_rate(buf),
        "pair_dt_p50_us": None,
        "pair_dt_p95_us": None,
        "pair_dt_p99_us": None,
        "pair_dt_max_us": None,
        "gap_p95_us": None,
        "gap_max_us": None,
        "ts_reversal_count": 0,
        "ts_min_signed_gap_us": None,
    }
    if n == 0:
        return summary

    # ── duration / rate / capture_ratio + timestamp gap (Rx1 기준 timestamp_us) ──
    ts = [
        row.get("timestamp_us")
        for row in buf
        if row.get("timestamp_us") is not None
    ]
    if len(ts) >= 2:
        duration_s = (max(ts) - min(ts)) / 1e6
        summary["duration_s"] = duration_s
        if duration_s > 0:
            rate = n / duration_s
            summary["pair_rate_hz"] = rate
            summary["capture_ratio"] = rate / CAPTURE_TARGET_HZ
        # 연속 패킷 간격(gap). abs 통계와 timestamp 역행을 분리해서 남긴다.
        signed_gaps = np.diff(np.asarray(ts, dtype=float))
        if signed_gaps.size:
            gaps_abs = np.abs(signed_gaps).tolist()
            summary["gap_p95_us"] = _percentile(gaps_abs, 95)
            summary["gap_max_us"] = float(max(gaps_abs))
            summary["ts_reversal_count"] = int(np.sum(signed_gaps < 0))
            summary["ts_min_signed_gap_us"] = float(signed_gaps.min())

    # ── pair_dt (legacy 버퍼엔 키 없음 → None 유지) ──
    pair_dts = [
        row["pair_dt_us"]
        for row in buf
        if row.get("pair_dt_us") is not None
    ]
    if pair_dts:
        summary["pair_dt_p50_us"] = _percentile(pair_dts, 50)
        summary["pair_dt_p95_us"] = _percentile(pair_dts, 95)
        summary["pair_dt_p99_us"] = _percentile(pair_dts, 99)
        summary["pair_dt_max_us"] = float(max(pair_dts))

    return summary


def _fmt_ms(value_us: Optional[float]) -> str:
    if value_us is None:
        return "N/A"
    return f"{value_us / 1000.0:.1f}ms"


def format_quality_lines(summary: dict) -> list[str]:
    """CLI/로그용 사람이 읽는 요약 라인 (report-only)."""
    return [
        "  pair_dt: "
        f"p50={_fmt_ms(summary['pair_dt_p50_us'])} "
        f"p95={_fmt_ms(summary['pair_dt_p95_us'])} "
        f"p99={_fmt_ms(summary['pair_dt_p99_us'])} "
        f"max={_fmt_ms(summary['pair_dt_max_us'])}",
        "  ts_gap : "
        f"p95={_fmt_ms(summary['gap_p95_us'])} "
        f"max={_fmt_ms(summary['gap_max_us'])} "
        f"reversal={summary.get('ts_reversal_count', 0)}",
    ]
