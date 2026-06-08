"""Replay static SafeSignal CSVs through the realtime inference path.

This is a demo preflight tool: it feeds stored CSV rows through
SlidingWindowBuffer and FallPredictor, then reports how many windows fired fall.

Examples
--------
    python debug/inference/replay_static_csv.py --limit-files 3
    set SAFESIGNAL_MODEL_PATH=model/finetune/checkpoints_item4_balanced/onset_primary_balanced_aug_s42/best_operating.pt
    set SAFESIGNAL_FALL_THRESHOLD=0.15
    set SAFESIGNAL_ENERGY_GATE_ENABLED=1
    set SAFESIGNAL_ENERGY_GATE_THRESHOLD=0.02
    python debug/inference/replay_static_csv.py --patterns "*E3*S03*NO_MOTION*.csv"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "server") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "server"))

from model.preprocessing.loader import load_safesignal_csv  # noqa: E402
from server.inference.buffer import SlidingWindowBuffer  # noqa: E402
from server.inference.predictor import FallPredictor  # noqa: E402


DEFAULT_PATTERNS = ("*NO_MOTION*.csv", "*STAND*.csv", "*LIE*.csv")


def iter_paths(root: Path, patterns: list[str], limit_files: int | None) -> list[Path]:
    seen: dict[str, Path] = {}
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                seen[str(p.resolve())] = p
    paths = list(seen.values())
    if limit_files is not None:
        paths = paths[:limit_files]
    return paths


def replay_file(path: Path, predictor: FallPredictor, limit_windows: int | None = None) -> dict:
    raw = load_safesignal_csv(path, rx="both")
    amp = raw.amplitude
    ts = raw.timestamps_us
    if amp.ndim != 2 or amp.shape[1] != 104:
        raise ValueError(f"{path.name}: expected amplitude shape (N,104), got {amp.shape}")

    buf = SlidingWindowBuffer()
    n_windows = 0
    n_fall = 0
    n_raw_fire = 0
    n_energy_skip = 0
    first_fall_seq = None

    for i in range(amp.shape[0]):
        triggered = buf.add(
            amp[i, :52],
            amp[i, 52:],
            timestamp_us=int(ts[i]) if ts is not None and len(ts) > i else 0,
        )
        if not triggered:
            continue

        n_windows += 1
        result = predictor.predict(buf.get_window())
        if result.get("raw_is_fall"):
            n_raw_fire += 1
        if result.get("energy_gate", {}).get("skipped"):
            n_energy_skip += 1
        if result.get("is_fall"):
            n_fall += 1
            if first_fall_seq is None:
                first_fall_seq = i
        if limit_windows is not None and n_windows >= limit_windows:
            break

    return {
        "file": path.name,
        "rows": int(amp.shape[0]),
        "windows": n_windows,
        "raw_fall_windows": n_raw_fire,
        "final_fall_windows": n_fall,
        "energy_gate_skips": n_energy_skip,
        "first_fall_row": first_fall_seq,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay static CSVs through server inference.")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    ap.add_argument("--patterns", nargs="+", default=list(DEFAULT_PATTERNS))
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--limit-windows-per-file", type=int, default=None)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "debug" / "inference" / "static_replay_report.json")
    args = ap.parse_args()

    paths = iter_paths(args.root, args.patterns, args.limit_files)
    if not paths:
        raise SystemExit(f"No CSV files matched {args.patterns} under {args.root}")

    predictor = FallPredictor()
    rows = []
    for p in paths:
        rec = replay_file(p, predictor, limit_windows=args.limit_windows_per_file)
        rows.append(rec)
        print(
            f"{p.name}: windows={rec['windows']} raw_fall={rec['raw_fall_windows']} "
            f"final_fall={rec['final_fall_windows']} energy_skip={rec['energy_gate_skips']}"
        )

    summary = {
        "n_files": len(rows),
        "total_windows": int(sum(r["windows"] for r in rows)),
        "total_raw_fall_windows": int(sum(r["raw_fall_windows"] for r in rows)),
        "total_final_fall_windows": int(sum(r["final_fall_windows"] for r in rows)),
        "total_energy_gate_skips": int(sum(r["energy_gate_skips"] for r in rows)),
        "files": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
