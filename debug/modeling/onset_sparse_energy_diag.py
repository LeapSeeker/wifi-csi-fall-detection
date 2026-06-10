"""Q-onset precheck: clean-400 full-session RPCA sparse frame energy.

Read-only diagnostic for deciding whether fall onset can be proposed from a
frame-wise spike. This script does not tune detector thresholds and does not
modify caches, model files, or training code.

Pipeline:
  SafeSignal CSV -> timestamp 100Hz resample -> first 550 frames
  -> remove stage beeps [0:50], [150:200], [400:450]
  -> clean400 = original[50:150] + [200:400] + [450:550]
  -> RPCA sparse on clean400 -> frame energy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model.preprocessing.loader import load_safesignal_csv, parse_safesignal_filename
from model.preprocessing.resample import resample_to_100hz
from model.preprocessing.rpca import DEFAULT_MAX_ITER, rpca_sparse

OUT_DIR = PROJECT_ROOT / "debug" / "modeling" / "diag_out" / "onset_sparse_energy"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"

FALL_ACTIVITIES = [
    "FALL_SIT_F",
    "FALL_SIT_B",
    "FALL_STD_F",
    "FALL_STD_B",
    "FALL_WALK_F",
    "FALL_WALK_B",
]
NONFALL_ACTIVITIES = ["WALK", "STAND", "SIT_STD"]

CLEAN_PRE = slice(0, 100)
CLEAN_FALL = slice(100, 300)
CLEAN_POST = slice(300, 400)
NOMINAL_ONSET = 100


def clean400_from_resampled(amp: np.ndarray) -> np.ndarray:
    if amp.shape[0] < 550:
        raise ValueError(f"resampled_count < 550: {amp.shape[0]}")
    base = amp[:550]
    return np.concatenate([base[50:150], base[200:400], base[450:550]], axis=0)


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(x, kernel, mode="same")


def robust_rise_onset(curve: np.ndarray, peak_idx: int) -> int:
    """First robust pre-peak rise crossing; exploratory, not a tuned detector."""
    base = curve[CLEAN_PRE]
    med = float(np.median(base))
    mad = float(np.median(np.abs(base - med)))
    thresh = med + 3.0 * 1.4826 * mad
    if mad <= 1e-12:
        thresh = med + 0.25 * (float(curve[peak_idx]) - med)
    lo = max(0, peak_idx - 120)
    hi = peak_idx + 1
    for i in range(lo, hi):
        if curve[i] >= thresh and np.all(curve[i : min(i + 5, curve.size)] >= thresh):
            return int(i)
    return int(peak_idx)


def discover_files(data_dir: Path, activities: list[str], per_activity: int, seed: int) -> list[Path]:
    by_activity: dict[str, list[Path]] = {a: [] for a in activities}
    for path in sorted(data_dir.glob("*.csv")):
        try:
            meta = parse_safesignal_filename(path)
        except ValueError:
            continue
        act = meta.activity.upper()
        if act in by_activity:
            by_activity[act].append(path)

    rng = np.random.default_rng(seed)
    selected: list[Path] = []
    for act in activities:
        pool = by_activity[act]
        if len(pool) <= per_activity:
            selected.extend(pool)
            continue
        idx = rng.choice(len(pool), size=per_activity, replace=False)
        selected.extend(pool[int(i)] for i in sorted(idx))
    return sorted(selected, key=lambda p: p.name)


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "p90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def process_file(path: Path, rpca_max_iter: int, smooth_width: int) -> tuple[dict, np.ndarray]:
    meta = parse_safesignal_filename(path)
    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us, target_hz=100.0, max_gap_ms=100.0)
    clean = clean400_from_resampled(res.amplitude)

    sparse = rpca_sparse(clean, max_iter=rpca_max_iter, tol=None)
    frame_energy = np.mean(np.abs(sparse), axis=1).astype(np.float64)
    curve = smooth(frame_energy, smooth_width)

    peak_idx = int(np.argmax(curve))
    rise_idx = robust_rise_onset(curve, peak_idx)
    pre_med = float(np.median(curve[CLEAN_PRE]))
    fall_peak = float(np.max(curve[CLEAN_FALL]))
    post_med = float(np.median(curve[CLEAN_POST]))
    global_peak = float(curve[peak_idx])
    pre_p95 = float(np.percentile(curve[CLEAN_PRE], 95))
    post_p95 = float(np.percentile(curve[CLEAN_POST], 95))
    denom = max(pre_med, post_med, 1e-12)

    cls = "fall" if meta.activity.upper().startswith("FALL") else "nonfall"
    rec = {
        "file": path.name,
        "activity": meta.activity,
        "class": cls,
        "env": meta.environment,
        "subject": meta.subject,
        "trial": meta.trial,
        "resampled_count": res.resampled_count,
        "original_rate_hz": res.original_rate_hz,
        "max_gap_ms": res.max_gap_us / 1000.0,
        "gap_count": res.gap_count,
        "peak_idx_clean": peak_idx,
        "rise_idx_clean": rise_idx,
        "peak_offset_from_nominal": peak_idx - NOMINAL_ONSET,
        "rise_offset_from_nominal": rise_idx - NOMINAL_ONSET,
        "peak_in_fall_range": int(100 <= peak_idx < 300),
        "rise_in_nominal_window_60_180": int(60 <= rise_idx <= 180),
        "pre_median": pre_med,
        "pre_p95": pre_p95,
        "fall_peak": fall_peak,
        "post_median": post_med,
        "post_p95": post_p95,
        "global_peak": global_peak,
        "fall_peak_over_static_median": float(fall_peak / denom),
        "global_peak_over_static_median": float(global_peak / denom),
    }
    return rec, curve


def maybe_plot(curves: dict[str, np.ndarray], rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    for cls in ("fall", "nonfall"):
        subset = [r for r in rows if r["class"] == cls][:12]
        if not subset:
            continue
        plt.figure(figsize=(10, 5))
        for r in subset:
            y = curves[r["file"]]
            plt.plot(y, alpha=0.65, lw=1.0, label=r["activity"] if len(subset) <= 6 else None)
        plt.axvspan(0, 100, color="#9aa0a6", alpha=0.12)
        plt.axvspan(100, 300, color="#d93025", alpha=0.10)
        plt.axvspan(300, 400, color="#188038", alpha=0.10)
        plt.axvline(100, color="black", ls="--", lw=0.8)
        plt.title(f"{cls} clean400 RPCA sparse frame energy")
        plt.xlabel("clean frame")
        plt.ylabel("mean abs sparse")
        if len(subset) <= 6:
            plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / f"{cls}_sample_curves.png", dpi=140)
        plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--per-activity", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rpca-max-iter", type=int, default=DEFAULT_MAX_ITER)
    ap.add_argument("--smooth-width", type=int, default=5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    activities = FALL_ACTIVITIES + NONFALL_ACTIVITIES
    files = discover_files(args.data_dir, activities, args.per_activity, args.seed)

    rows: list[dict] = []
    curves: dict[str, np.ndarray] = {}
    skipped: list[dict] = []
    for i, path in enumerate(files, start=1):
        try:
            rec, curve = process_file(path, args.rpca_max_iter, args.smooth_width)
        except Exception as exc:
            skipped.append({"file": path.name, "error": str(exc)})
            continue
        rows.append(rec)
        curves[path.name] = curve
        print(f"[{i:03d}/{len(files):03d}] {path.name} peak={rec['peak_idx_clean']} rise={rec['rise_idx_clean']} "
              f"ratio={rec['fall_peak_over_static_median']:.2f}")

    csv_path = OUT_DIR / "onset_sparse_energy_windows.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    by_class: dict[str, list[dict]] = defaultdict(list)
    by_activity: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)
        by_activity[r["activity"]].append(r)

    summary: dict = {
        "note": "Exploratory Q-onset precheck only; no detector threshold tuning.",
        "coordinate": "clean400 = original[50:150] + [200:400] + [450:550] after 100Hz resample and [:550]",
        "rpca_max_iter": args.rpca_max_iter,
        "smooth_width": args.smooth_width,
        "selected_files": len(files),
        "processed": len(rows),
        "skipped": skipped,
        "by_class": {},
        "by_activity": {},
    }

    metric_names = [
        "global_peak",
        "fall_peak",
        "fall_peak_over_static_median",
        "global_peak_over_static_median",
        "peak_idx_clean",
        "rise_idx_clean",
        "peak_offset_from_nominal",
        "rise_offset_from_nominal",
    ]
    for key, vals in by_class.items():
        summary["by_class"][key] = {m: quantiles([float(r[m]) for r in vals]) for m in metric_names}
        summary["by_class"][key]["peak_in_fall_range_rate"] = (
            float(np.mean([r["peak_in_fall_range"] for r in vals])) if vals else None
        )
        summary["by_class"][key]["rise_in_nominal_window_60_180_rate"] = (
            float(np.mean([r["rise_in_nominal_window_60_180"] for r in vals])) if vals else None
        )
    for key, vals in by_activity.items():
        summary["by_activity"][key] = {
            "n": len(vals),
            "global_peak": quantiles([float(r["global_peak"]) for r in vals]),
            "fall_peak_over_static_median": quantiles(
                [float(r["fall_peak_over_static_median"]) for r in vals]
            ),
            "peak_idx_clean": quantiles([float(r["peak_idx_clean"]) for r in vals]),
            "rise_idx_clean": quantiles([float(r["rise_idx_clean"]) for r in vals]),
        }

    fall_peaks = [float(r["global_peak"]) for r in rows if r["class"] == "fall"]
    nonfall_rows = [r for r in rows if r["class"] == "nonfall"]
    if fall_peaks and nonfall_rows:
        for q in (50, 75, 90):
            thr = float(np.percentile(fall_peaks, q))
            hits = [r for r in nonfall_rows if float(r["global_peak"]) >= thr]
            summary[f"nonfall_ge_fall_p{q}"] = {
                "threshold": thr,
                "n": len(hits),
                "rate": len(hits) / len(nonfall_rows),
                "by_activity": {
                    act: sum(1 for r in hits if r["activity"] == act)
                    for act in sorted({r["activity"] for r in nonfall_rows})
                },
            }

    maybe_plot(curves, rows, OUT_DIR)
    with (OUT_DIR / "onset_sparse_energy_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[out] {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
