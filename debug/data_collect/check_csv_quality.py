"""data/raw/ 폴더 내 수집된 CSV 파일 일괄 품질 검증.

파일명 규칙: E{env}_S{subj:02d}_A_{act}_T{trial:03d}.csv
컬럼 구조:
  - 필수(107컬럼 공통): timestamp_us, seq_rx1, seq_rx2, amp_rx1_0..51, amp_rx2_0..51
  - 신규(110컬럼): + timestamp_rx1_us, timestamp_rx2_us, pair_dt_us (페어링 품질용)
컬럼 수가 아니라 필수 컬럼명 존재로 검증하므로 107/110 CSV를 모두 통과시킨다.
pair_dt_us가 있으면 p50/p95/p99/max를 출력하고, 없으면 legacy로 보고 N/A 표시한다.
pair_dt/gap은 report-only — RECOLLECT 판정과 강하게 연결하지 않는다.

실행:
    python debug/data_collect/check_csv_quality.py
    python debug/data_collect/check_csv_quality.py --dir data/raw/
    python debug/data_collect/check_csv_quality.py --fail_only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# 활동별 최소 기대 row 수 (100Hz 기준, duration × 100Hz × 80%)
MIN_ROWS: dict[str, int] = {
    "WALK":      640,   # 8s × 100Hz × 80%
    "RUN":       640,
    "STAND":     400,   # 5s × 100Hz × 80%
    "PICK":      400,
    "LIE":       560,   # 7s × 100Hz × 80%
    "SIT_STD":   480,   # 6s × 100Hz × 80%
    "FALL_SIT_F": 400,  # 5s × 100Hz × 80%
    "FALL_SIT_B": 400,
    "FALL_STD_F": 400,
    "FALL_STD_B": 400,
    "FALL_WALK_F": 400,
    "FALL_WALK_B": 400,
}
DEFAULT_MIN_ROWS = 300

# 컬럼 수 대신 필수 컬럼명 존재로 검증 (107/110 CSV 공통 통과)
REQUIRED_COLS: list[str] = (
    ["timestamp_us", "seq_rx1", "seq_rx2"]
    + [f"amp_rx1_{i}" for i in range(52)]
    + [f"amp_rx2_{i}" for i in range(52)]
)


def _percentile(values, q):
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _fmt_ms(value_us):
    if value_us is None:
        return "N/A"
    return f"{value_us / 1000.0:.1f}ms"


def calc_loss_rate(seqs: list[int]) -> float:
    if len(seqs) <= 1:
        return 0.0
    lo, hi = min(seqs), max(seqs)
    expected = hi - lo + 1
    if expected <= 0:
        return 0.0
    unique = len(set(seqs))
    return (expected - unique) / expected


def check_seq_monotonic(seqs: list[int]) -> bool:
    for i in range(1, len(seqs)):
        if seqs[i] < seqs[i - 1]:
            return False
    return True


def parse_filename(fname: str) -> dict | None:
    pattern = r"^E(\d+)_S(\d+)_A_(.+)_T(\d+)\.csv$"
    m = re.match(pattern, fname)
    if not m:
        return None
    return {
        "env": int(m.group(1)),
        "subject": int(m.group(2)),
        "activity": m.group(3),
        "trial": int(m.group(4)),
    }


def check_file(csv_path: Path) -> dict:
    result = {
        "file": csv_path.name,
        "rows": 0,
        "cols": 0,
        "loss_rx1": 0.0,
        "loss_rx2": 0.0,
        "seq_ok": True,
        "neg_ratio": 0.0,
        "zero_ratio": 0.0,
        "status": "OK",
        "warnings": [],
        # 페어링 품질 (report-only). pair_dt는 legacy(107) CSV면 None.
        "pair_dt_p50_us": None,
        "pair_dt_p95_us": None,
        "pair_dt_p99_us": None,
        "pair_dt_max_us": None,
        "gap_p95_us": None,
        "gap_max_us": None,
    }

    meta = parse_filename(csv_path.name)
    if meta is None:
        result["status"] = "SKIP"
        result["warnings"].append("파일명 규칙 불일치")
        return result

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result["status"] = "ERROR"
        result["warnings"].append(f"읽기 실패: {e}")
        return result

    result["rows"] = len(df)
    result["cols"] = len(df.columns)

    # 필수 컬럼명 존재 검증 (컬럼 수 무관 — 107/110 모두 허용)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        result["status"] = "ERROR"
        head = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        result["warnings"].append(f"필수 컬럼 누락({len(missing)}개): {head}")

    # row 수 검증
    activity = meta["activity"]
    min_rows = MIN_ROWS.get(activity, DEFAULT_MIN_ROWS)
    if result["rows"] < min_rows:
        result["warnings"].append(f"row 수 {result['rows']} < 최소 {min_rows}")

    # seq 손실률 / 단조성
    if "seq_rx1" in df.columns and "seq_rx2" in df.columns:
        seq_rx1 = df["seq_rx1"].tolist()
        seq_rx2 = df["seq_rx2"].tolist()
        result["loss_rx1"] = calc_loss_rate(seq_rx1)
        result["loss_rx2"] = calc_loss_rate(seq_rx2)
        result["seq_ok"] = check_seq_monotonic(seq_rx1) and check_seq_monotonic(seq_rx2)

        if not result["seq_ok"]:
            result["warnings"].append("seq 역전 발생")
        if result["loss_rx1"] >= 0.20 or result["loss_rx2"] >= 0.20:
            result["status"] = "RECOLLECT"
            result["warnings"].append(f"손실률 ≥20% — 재수집 필요")
        elif result["loss_rx1"] >= 0.05 or result["loss_rx2"] >= 0.05:
            if result["status"] == "OK":
                result["status"] = "WARN"
            result["warnings"].append("손실률 ≥5%")

    # 진폭 이상값
    amp_cols = [c for c in df.columns if c.startswith("amp_")]
    if amp_cols:
        amp = df[amp_cols].values.astype(np.float32)
        result["neg_ratio"] = float((amp < 0).mean())
        result["zero_ratio"] = float((amp == 0).mean())

    # pair delay (신규 110컬럼 CSV만; legacy면 None 유지) — report-only
    if "pair_dt_us" in df.columns:
        pair_dt = df["pair_dt_us"].dropna().to_numpy(dtype=float)
        if pair_dt.size:
            result["pair_dt_p50_us"] = _percentile(pair_dt, 50)
            result["pair_dt_p95_us"] = _percentile(pair_dt, 95)
            result["pair_dt_p99_us"] = _percentile(pair_dt, 99)
            result["pair_dt_max_us"] = float(pair_dt.max())

    # timestamp gap (timestamp_us diff 기준) — report-only
    if "timestamp_us" in df.columns and len(df) >= 2:
        gaps = np.abs(np.diff(df["timestamp_us"].to_numpy(dtype=float)))
        if gaps.size:
            result["gap_p95_us"] = _percentile(gaps, 95)
            result["gap_max_us"] = float(gaps.max())

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="수집 CSV 품질 검증")
    parser.add_argument("--dir", type=str, default="data/raw", help="CSV 폴더 경로")
    parser.add_argument("--fail_only", action="store_true", help="기준 미달만 출력")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.dir
    if not data_dir.exists():
        print(f"[ERROR] 폴더 없음: {data_dir}")
        sys.exit(1)

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        print(f"[INFO] CSV 파일 없음: {data_dir}")
        sys.exit(0)

    print(f"=== CSV 품질 검증 ({len(csvs)}개 파일) ===\n")
    print(f"  {'파일':<40}  {'rows':>5}  {'cols':>5}  "
          f"{'loss_rx1':>9}  {'loss_rx2':>9}  {'seq_ok':>7}  {'status':>9}")
    print("  " + "-" * 100)

    counts = {"OK": 0, "WARN": 0, "RECOLLECT": 0, "ERROR": 0, "SKIP": 0}

    for csv_path in csvs:
        r = check_file(csv_path)
        counts[r["status"]] = counts.get(r["status"], 0) + 1

        if args.fail_only and r["status"] == "OK":
            continue

        status_color = r["status"]
        print(f"  {r['file']:<40}  {r['rows']:>5}  {r['cols']:>5}  "
              f"  {r['loss_rx1']*100:>7.1f}%  {r['loss_rx2']*100:>7.1f}%  "
              f"{'OK' if r['seq_ok'] else 'FAIL':>7}  {status_color:>9}")

        # 페어링 품질 (report-only). pair_dt는 legacy CSV면 N/A.
        if r["pair_dt_p50_us"] is None:
            print(f"    └ pair_dt: N/A (legacy 107컬럼)")
        else:
            print(
                f"    └ pair_dt: p50={_fmt_ms(r['pair_dt_p50_us'])} "
                f"p95={_fmt_ms(r['pair_dt_p95_us'])} "
                f"p99={_fmt_ms(r['pair_dt_p99_us'])} "
                f"max={_fmt_ms(r['pair_dt_max_us'])}"
            )
        print(
            f"    └ ts_gap : p95={_fmt_ms(r['gap_p95_us'])} "
            f"max={_fmt_ms(r['gap_max_us'])}"
        )

        for w in r["warnings"]:
            print(f"    └ {w}")

    print("\n  " + "-" * 100)
    print(f"\n[요약]")
    print(f"  총 파일: {len(csvs)}")
    print(f"  OK: {counts.get('OK', 0)}  "
          f"경고: {counts.get('WARN', 0)}  "
          f"재수집 필요: {counts.get('RECOLLECT', 0)}  "
          f"오류: {counts.get('ERROR', 0)}")
    print("\nDONE")


if __name__ == "__main__":
    main()
