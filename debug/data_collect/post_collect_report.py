"""수집 직후 완료/재수집 후보를 빠르게 요약한다.

권장 흐름:
    python tools/safesignal_debug.py clean-csv --env E4 --overwrite
    python tools/safesignal_debug.py post-collect --env E4 --dir data/cleaned

판정 기준은 timestamp 정렬 후 기준(Q3)만 사용한다.
  - loss rate >= 10%
  - absolute timestamp gap p95 >= 30ms
  - absolute timestamp gap max >= 150ms
  - pair_dt p95 >= 25ms
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from collect.labels import ACTIVITY_INFO, ACTIVITY_ORDER
from debug.data_collect.check_csv_quality import check_file, parse_filename


LOSS_RECOLLECT = 0.10
GAP_P95_RECOLLECT_US = 30_000.0
GAP_MAX_RECOLLECT_US = 150_000.0
PAIR_P95_RECOLLECT_US = 25_000.0


def _fmt_ms(value_us: float | None) -> str:
    if value_us is None:
        return "N/A"
    return f"{value_us / 1000.0:.1f}ms"


def _q3_reasons(result: dict) -> list[str]:
    reasons: list[str] = []
    loss = max(float(result.get("loss_rx1") or 0.0), float(result.get("loss_rx2") or 0.0))
    if loss >= LOSS_RECOLLECT:
        reasons.append(f"loss={loss * 100:.1f}%")

    gap_p95 = result.get("gap_p95_us")
    if gap_p95 is not None and float(gap_p95) >= GAP_P95_RECOLLECT_US:
        reasons.append(f"gap_p95={_fmt_ms(float(gap_p95))}")

    gap_max = result.get("gap_max_us")
    if gap_max is not None and float(gap_max) >= GAP_MAX_RECOLLECT_US:
        reasons.append(f"gap_max={_fmt_ms(float(gap_max))}")

    pair_p95 = result.get("pair_dt_p95_us")
    if pair_p95 is not None and float(pair_p95) >= PAIR_P95_RECOLLECT_US:
        reasons.append(f"pair_p95={_fmt_ms(float(pair_p95))}")

    if result.get("status") == "ERROR":
        reasons.append("read_or_schema_error")
    return reasons


def _collect_records(
    data_dir: Path,
    env: int,
    subjects: set[int] | None,
    activities: set[str] | None,
) -> list[dict]:
    records: list[dict] = []
    for path in sorted(data_dir.glob(f"E{env}_S*_A_*.csv")):
        meta = parse_filename(path.name)
        if meta is None:
            continue
        if subjects is not None and int(meta["subject"]) not in subjects:
            continue
        activity = str(meta["activity"]).upper()
        if activities is not None and activity not in activities:
            continue
        result = check_file(path)
        records.append({
            "path": path,
            "file": path.name,
            "env": int(meta["env"]),
            "subject": int(meta["subject"]),
            "activity": activity,
            "trial": int(meta["trial"]),
            "quality": result,
            "q3_reasons": _q3_reasons(result),
        })
    return records


def _activity_target(activity: str, include_no_motion: bool) -> int:
    if activity == "NO_MOTION" and not include_no_motion:
        return 0
    return int(ACTIVITY_INFO[activity]["target"])


def _subjects(records: list[dict]) -> list[int]:
    return sorted({int(r["subject"]) for r in records})


def _missing_trials(records: list[dict], subject: int, activity: str, target: int) -> list[int]:
    existing = {
        int(r["trial"])
        for r in records
        if int(r["subject"]) == subject and r["activity"] == activity
    }
    return [trial for trial in range(1, target + 1) if trial not in existing]


def _write(lines: list[str], text: str = "") -> None:
    lines.append(text)
    print(text)


def build_report(
    records: list[dict],
    env: int,
    include_no_motion: bool,
    show_ok: bool,
    expected_subjects: list[int] | None = None,
    activities: list[str] | None = None,
) -> str:
    lines: list[str] = []
    subjects = sorted(expected_subjects) if expected_subjects else _subjects(records)
    activity_order = [a.upper() for a in activities] if activities else ACTIVITY_ORDER

    _write(lines, f"=== E{env} post-collection report ===")
    _write(lines, f"files={len(records)} subjects={','.join(f'S{s:02d}' for s in subjects) or 'none'}")
    if activities:
        _write(lines, f"activities={','.join(activity_order)}")
    _write(lines)

    _write(lines, "[completion]")
    incomplete: list[str] = []
    for subject in subjects:
        _write(lines, f"  S{subject:02d}")
        for activity in activity_order:
            if activity not in ACTIVITY_INFO:
                _write(lines, f"    {activity:<12} SKIP unknown_activity")
                continue
            target = _activity_target(activity, include_no_motion)
            if target == 0:
                continue
            count = sum(
                1
                for r in records
                if int(r["subject"]) == subject and r["activity"] == activity
            )
            missing = _missing_trials(records, subject, activity, target)
            status = "OK" if count >= target and not missing else "MISSING"
            if status == "MISSING":
                miss_txt = ",".join(f"T{t:03d}" for t in missing)
                incomplete.append(f"E{env}_S{subject:02d}_{activity}: {miss_txt}")
            elif not show_ok:
                miss_txt = ""
            else:
                miss_txt = "complete"
            if show_ok or status == "MISSING":
                _write(lines, f"    {activity:<12} {count:>2}/{target:<2} {status:<7} {miss_txt}")
    if not subjects:
        _write(lines, "  no files")
    _write(lines)

    recollect = [r for r in records if r["q3_reasons"]]
    _write(lines, "[q3_recollect_candidates]")
    if recollect:
        for r in sorted(recollect, key=lambda x: (x["subject"], x["activity"], x["trial"])):
            reasons = ", ".join(r["q3_reasons"])
            q = r["quality"]
            _write(
                lines,
                f"  {r['file']}  reasons=[{reasons}] "
                f"rows={q['rows']} gap_max={_fmt_ms(q.get('gap_max_us'))} "
                f"pair_p95={_fmt_ms(q.get('pair_dt_p95_us'))}",
            )
    else:
        _write(lines, "  none")
    _write(lines)

    _write(lines, "[next_actions]")
    if incomplete:
        _write(lines, "  1. Missing trial 먼저 채우기")
        for item in incomplete:
            _write(lines, f"     - {item}")
    else:
        _write(lines, "  1. Missing trial 없음")
    if recollect:
        _write(lines, "  2. q3_recollect_candidates 파일은 재수집 또는 격리 판단")
    else:
        _write(lines, "  2. Q3 기준 재수집 후보 없음")
    _write(lines, "  3. 최종 수집 후 clean-csv 재실행")
    _write(lines, "  4. 모델 학습/평가에는 data/cleaned 기준 사용")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="수집 직후 완료/품질 요약")
    parser.add_argument("--env", required=True, type=int, help="환경 번호 예: 4")
    parser.add_argument("--dir", default="data/cleaned", help="CSV 폴더")
    parser.add_argument("--subject", nargs="+", type=int, default=None, help="대상 subject 예: 2 3")
    parser.add_argument("--activity", nargs="+", default=None, help="대상 활동 예: WALK SIT_STD FALL_SIT_F")
    parser.add_argument("--include-no-motion", action="store_true", help="NO_MOTION target도 완료율에 포함")
    parser.add_argument("--show-ok", action="store_true", help="완료된 활동도 모두 출력")
    parser.add_argument("--out", default=None, help="리포트 저장 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = (PROJECT_ROOT / args.dir).resolve()
    if not data_dir.exists():
        raise SystemExit(f"[ERROR] CSV folder not found: {data_dir}")

    subjects = set(args.subject) if args.subject else None
    activities = {str(a).upper() for a in args.activity} if args.activity else None
    records = _collect_records(data_dir, args.env, subjects, activities)
    report = build_report(
        records,
        args.env,
        args.include_no_motion,
        args.show_ok,
        expected_subjects=sorted(subjects) if subjects else None,
        activities=sorted(activities) if activities else None,
    )

    if args.out:
        out = (PROJECT_ROOT / args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
