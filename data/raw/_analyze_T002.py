"""T002 CSV 1회성 분석 스크립트 (자체수집 데이터 sanity check)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).with_name("E1_S01_A_STAND_T001.csv")

df = pd.read_csv(CSV)
n = len(df)

ts = df["timestamp_us"].to_numpy()
seq1 = df["seq_rx1"].to_numpy()
seq2 = df["seq_rx2"].to_numpy()
amp1 = df[[f"amp_rx1_{i}" for i in range(52)]].to_numpy()
amp2 = df[[f"amp_rx2_{i}" for i in range(52)]].to_numpy()

print(f"=== File: {CSV.name} ===")
print(f"rows                : {n}")
print(f"columns             : {len(df.columns)}")
print()

# 1. Duration / sampling rate (host pairing timestamp)
dur_s = (ts.max() - ts.min()) / 1e6
print(f"duration (host ts)  : {dur_s:.3f} s")
print(f"avg rate (rows/s)   : {n / dur_s:.2f}")

dts = np.diff(np.sort(ts)) / 1e3  # ms
print(f"row dt   (ms)       : mean={dts.mean():.2f}  median={np.median(dts):.2f}  "
      f"p95={np.percentile(dts,95):.2f}  max={dts.max():.2f}")
print()

# 2. Sequence-based loss
def loss_rate(seq: np.ndarray) -> tuple[int, int, float]:
    span = int(seq.max() - seq.min() + 1)
    uniq = int(np.unique(seq).size)
    return span, uniq, (span - uniq) / span

for name, s in [("rx1", seq1), ("rx2", seq2)]:
    span, uniq, lr = loss_rate(s)
    print(f"{name} seq span        : {span}  unique={uniq}  loss={lr*100:.2f}%")
print()

# 3. seq monotonicity
print(f"rx1 seq monotonic   : {bool(np.all(np.diff(seq1) >= 0))}  "
      f"non-mono={int((np.diff(seq1) < 0).sum())}")
print(f"rx2 seq monotonic   : {bool(np.all(np.diff(seq2) >= 0))}  "
      f"non-mono={int((np.diff(seq2) < 0).sum())}")
print()

# 4. timestamp_us monotonicity (페어링 결과는 더 흐트러질 수 있음)
print(f"timestamp_us monotonic : {bool(np.all(np.diff(ts) >= 0))}  "
      f"non-mono={int((np.diff(ts) < 0).sum())}")
print()

# 5. Pairing skew (rx1<->rx2 같은 row의 seq 차이는 고정 오프셋이어야 자연스러움)
seq_off = seq1.astype(np.int64) - seq2.astype(np.int64)
print(f"seq1-seq2 offset    : mean={seq_off.mean():.2f}  std={seq_off.std():.2f}  "
      f"min={seq_off.min()}  max={seq_off.max()}")
print()

# 6. Amplitude statistics
def amp_stats(name: str, amp: np.ndarray) -> None:
    flat = amp.reshape(-1)
    nz = (flat == 0).sum()
    print(f"{name}: shape={amp.shape}  min={flat.min():.2f}  max={flat.max():.2f}  "
          f"mean={flat.mean():.2f}  std={flat.std():.2f}  zeros={nz} ({nz/flat.size*100:.1f}%)")

amp_stats("rx1 amplitude       ", amp1)
amp_stats("rx2 amplitude       ", amp2)
print()

# 7. Per-subcarrier zero fraction (DC/guard 식별)
zero_frac1 = (amp1 == 0).mean(axis=0)
zero_frac2 = (amp2 == 0).mean(axis=0)
zero_idx1 = np.where(zero_frac1 > 0.5)[0]
zero_idx2 = np.where(zero_frac2 > 0.5)[0]
print(f"rx1 subcarriers >50% zero : {zero_idx1.tolist()}")
print(f"rx2 subcarriers >50% zero : {zero_idx2.tolist()}")
print()

# 8. Per-subcarrier std (활성 subcarrier 식별 + 변화량)
std1 = amp1.std(axis=0)
std2 = amp2.std(axis=0)
active1 = np.where(std1 > 0.1)[0]
active2 = np.where(std2 > 0.1)[0]
print(f"rx1 active subcarriers (std>0.1) : {active1.size} / 52")
print(f"rx2 active subcarriers (std>0.1) : {active2.size} / 52")
print(f"rx1 std mean (active)            : {std1[active1].mean():.3f}  max={std1.max():.3f}")
print(f"rx2 std mean (active)            : {std2[active2].mean():.3f}  max={std2.max():.3f}")
print()

# 9. RSSI(있으면)
rssi_cols = [c for c in df.columns if "rssi" in c.lower()]
print(f"rssi columns        : {rssi_cols}")

# 10. timestamp_us reality check (Unix time 2026 ≈ 1.74e15 µs)
print()
print(f"first ts            : {ts[0]}  (={ts[0]/1e6:.0f} sec since epoch)")
import datetime as dt
try:
    print(f"first ts as UTC     : {dt.datetime.utcfromtimestamp(ts[0]/1e6)}")
except Exception as e:
    print(f"first ts decode err : {e}")
