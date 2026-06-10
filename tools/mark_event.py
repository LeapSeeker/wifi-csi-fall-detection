"""Append a manual wall-clock event marker for demo diagnostics.

Usage examples:
  python tools/mark_event.py fall_start
  python tools/mark_event.py impact "backward fall"
  python tools/mark_event.py alert_seen
  python tools/mark_event.py false_alarm
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "server" / "logs"
EVENT_FILE = LOG_DIR / "manual_events.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a manual diagnostic event marker.")
    parser.add_argument(
        "event",
        help="Event name, e.g. static_start, fall_start, impact, alert_seen, false_alarm.",
    )
    parser.add_argument("note", nargs="?", default="", help="Optional short note.")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    exists = EVENT_FILE.exists()
    now = dt.datetime.now()
    row = {
        "wall": now.strftime("%H:%M:%S.%f")[:-3],
        "iso": now.isoformat(timespec="milliseconds"),
        "event": args.event,
        "note": args.note,
    }
    with EVENT_FILE.open("a", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=["wall", "iso", "event", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[manual-event] {row['wall']} {args.event} {args.note}".rstrip())
    print(f"[manual-event] saved: {EVENT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
