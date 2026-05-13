"""SafeSignal 자체수집 전처리 스모크 테스트 (D-018).

실행:
    python -m model.preprocessing.test_safesignal
또는:
    python model/preprocessing/test_safesignal.py

검증 항목:
  1) load_safesignal_csv(rx="both") shape == (n, 104)
  2) resample_to_100hz: timestamps 간격이 정확히 10,000us, 외삽 없음
  3) preprocess_safesignal_file(tail_window=True) → windows shape (?, 300, 104)
  4) (optional, 환경변수 SAFESIGNAL_FULL_TEST=1) preprocess_safesignal_file_full
     → inputs shape (?, 1, 28, 20). RPCA가 무거우므로 기본 OFF.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from model.preprocessing import (
    WINDOW_SIZE,
    load_safesignal_csv,
    parse_safesignal_filename,
    preprocess_safesignal_file,
    preprocess_safesignal_file_full,
    resample_to_100hz,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CSV = REPO_ROOT / "data" / "raw" / "E1_S01_A_STAND_T001.csv"
N_SC_EACH = 52


def _make_synthetic_csv(path: Path, n_packets: int = 400, rate_hz: float = 70.0) -> None:
    """파일명 규칙(E1_S99_A_STAND_T001.csv) + 107 컬럼 합성 CSV 생성."""
    step_us = int(round(1_000_000.0 / rate_hz))
    ts = np.arange(n_packets, dtype=np.int64) * step_us + 1_000_000
    rng = np.random.default_rng(0)
    cols = {"timestamp_us": ts, "seq_rx1": np.arange(n_packets), "seq_rx2": np.arange(n_packets)}
    for j in range(N_SC_EACH):
        cols[f"amp_rx1_{j}"] = (10.0 + rng.standard_normal(n_packets)).astype(np.float32)
    for j in range(N_SC_EACH):
        cols[f"amp_rx2_{j}"] = (15.0 + rng.standard_normal(n_packets)).astype(np.float32)
    pd.DataFrame(cols).to_csv(path, index=False)


def _hr(title: str) -> None:
    print(f"\n[{title}]")


def test_filename_parser() -> None:
    _hr("1) parse_safesignal_filename")
    meta = parse_safesignal_filename("E1_S01_A_STAND_T001.csv")
    assert meta.environment == 1, meta
    assert meta.subject == 1, meta
    assert meta.activity == "STAND", meta
    assert meta.trial == 1, meta
    print("  OK", meta)

    meta2 = parse_safesignal_filename("E2_S03_A_SIT_STD_T012.csv")
    assert meta2.activity == "SIT_STD", meta2
    print("  OK underscore-activity", meta2)


def test_loader_synthetic(tmp_path: Path) -> tuple[Path, int]:
    _hr("2) load_safesignal_csv (synthetic, rx='both')")
    csv = tmp_path / "E1_S99_A_STAND_T001.csv"
    n = 400
    _make_synthetic_csv(csv, n_packets=n, rate_hz=70.0)
    raw = load_safesignal_csv(csv, rx="both")
    assert raw.amplitude.shape == (n, 104), raw.amplitude.shape
    assert raw.timestamps_us.shape == (n,), raw.timestamps_us.shape
    assert raw.amplitude.dtype == np.float32
    assert raw.rx == "both"
    print(f"  OK shape={raw.amplitude.shape} dtype={raw.amplitude.dtype}")

    raw_rx1 = load_safesignal_csv(csv, rx="rx1")
    assert raw_rx1.amplitude.shape == (n, 52), raw_rx1.amplitude.shape
    print(f"  OK rx1 shape={raw_rx1.amplitude.shape}")

    return csv, n


def test_resample_grid(csv: Path) -> None:
    _hr("3) resample_to_100hz: 10,000us grid + no extrapolation")
    raw = load_safesignal_csv(csv, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us, target_hz=100.0)

    diffs = np.diff(res.timestamps_us)
    assert diffs.size > 0
    assert np.all(diffs == 10_000), f"non-uniform grid: unique diffs={np.unique(diffs)[:5]}"
    assert res.timestamps_us[0] >= raw.timestamps_us.min()
    assert res.timestamps_us[-1] <= raw.timestamps_us.max()
    assert res.amplitude.shape[1] == raw.amplitude.shape[1]
    assert res.target_hz == 100.0
    assert res.original_count == raw.amplitude.shape[0]
    print(
        f"  OK n_resampled={res.resampled_count} step={diffs[0]}us "
        f"orig_rate={res.original_rate_hz:.2f}Hz "
        f"max_gap={res.max_gap_us}us gap_count={res.gap_count}"
    )


def test_resample_handles_duplicates_and_disorder() -> None:
    _hr("4) resample_to_100hz: duplicate + out-of-order timestamps")
    # 10 samples, but with one duplicate and one swap
    ts = np.array(
        [0, 10_000, 10_000, 30_000, 20_000, 40_000, 50_000, 60_000, 70_000, 80_000],
        dtype=np.int64,
    )
    amp = np.tile(np.arange(ts.size, dtype=np.float32)[:, None], (1, 4))
    res = resample_to_100hz(amp, ts, target_hz=100.0)
    assert np.all(np.diff(res.timestamps_us) == 10_000)
    # 80,000us range / 10,000us = 8 → 8+1 = 9 grid points
    assert res.resampled_count == 9, res.resampled_count
    assert res.original_count == ts.size
    print(f"  OK resampled_count={res.resampled_count} (deduped + sorted)")


def test_pipeline_synthetic(csv: Path) -> None:
    _hr("5) preprocess_safesignal_file (tail_window=True)")
    pre = preprocess_safesignal_file(csv, rx="both", tail_window=True)
    assert pre.windows.ndim == 3
    assert pre.windows.shape[1] == WINDOW_SIZE
    assert pre.windows.shape[2] == 104
    assert pre.windows.shape[0] >= 1
    print(
        f"  OK windows={pre.windows.shape} "
        f"resample(orig={pre.resample.original_count}, "
        f"resampled={pre.resample.resampled_count}, "
        f"orig_rate={pre.resample.original_rate_hz:.2f}Hz)"
    )


def test_pipeline_real() -> None:
    _hr("6) preprocess_safesignal_file (real CSV, tail_window=True)")
    if not REAL_CSV.exists():
        print(f"  SKIP (no real CSV at {REAL_CSV})")
        return
    pre = preprocess_safesignal_file(REAL_CSV, rx="both", tail_window=True)
    assert pre.windows.shape[1:] == (WINDOW_SIZE, 104), pre.windows.shape
    assert pre.windows.shape[0] >= 1
    print(
        f"  OK {REAL_CSV.name} windows={pre.windows.shape} "
        f"orig_rate={pre.resample.original_rate_hz:.2f}Hz "
        f"max_gap={pre.resample.max_gap_us}us gap_count={pre.resample.gap_count}"
    )


def test_full_optional() -> None:
    if os.environ.get("SAFESIGNAL_FULL_TEST", "0") != "1":
        _hr("7) preprocess_safesignal_file_full (SKIP — set SAFESIGNAL_FULL_TEST=1 to run)")
        return
    _hr("7) preprocess_safesignal_file_full")
    target = REAL_CSV if REAL_CSV.exists() else None
    if target is None:
        print("  SKIP (no real CSV)")
        return
    res = preprocess_safesignal_file_full(
        target, rx="both", tail_window=True, rpca_max_iter=20
    )
    assert res.inputs.shape[1:] == (1, 28, 20), res.inputs.shape
    assert res.inputs.shape[0] >= 1
    print(f"  OK inputs={res.inputs.shape}")


def main() -> None:
    test_filename_parser()
    with tempfile.TemporaryDirectory() as td:
        csv, _ = test_loader_synthetic(Path(td))
        test_resample_grid(csv)
        test_resample_handles_duplicates_and_disorder()
        test_pipeline_synthetic(csv)
    test_pipeline_real()
    test_full_optional()
    print("\nALL_OK")


if __name__ == "__main__":
    main()
