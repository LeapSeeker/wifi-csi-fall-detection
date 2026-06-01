"""Augmentation verification gate (b안). Read-only check; does not modify train.py."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.finetune.train import (
    SOURCE_ALSAIFY,
    SOURCE_SAFESIGNAL,
    TrainAugmentDataset,
)


class StubSubset:
    """Minimal subset yielding (x, y, meta) with a fixed source tag."""

    def __init__(self, source: str) -> None:
        self.source = source
        rng = np.random.default_rng(0)
        self.X = torch.as_tensor(
            rng.standard_normal((4, 1, 28, 20)), dtype=torch.float32
        )

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        meta = {"subject": 1, "source": self.source, "env": 1, "filename": f"f{idx}"}
        return self.X[idx], torch.tensor(0, dtype=torch.long), meta


results = {}

# Gate 1: reproducibility — same (idx, epoch) twice => bit-identical
safe = StubSubset(SOURCE_SAFESIGNAL)
ds = TrainAugmentDataset(safe, augment=True, base_seed=42)
ds.set_epoch(3)
x1 = ds[0][0]
x2 = ds[0][0]
results["1_reproducibility"] = bool(torch.equal(x1, x2))

# Gate 2: diversity — same idx, epoch 0 vs 1 => different
ds.set_epoch(0)
xe0 = ds[0][0].clone()
ds.set_epoch(1)
xe1 = ds[0][0].clone()
results["2_diversity"] = bool(not torch.equal(xe0, xe1))

# Gate 3: Alsaify unchanged even with augment=True => bit-identical to original
als = StubSubset(SOURCE_ALSAIFY)
orig_als0 = als.X[0].clone()
ds_als = TrainAugmentDataset(als, augment=True, base_seed=42)
ds_als.set_epoch(0)
x_als = ds_als[0][0]
results["3_alsaify_unchanged"] = bool(torch.equal(x_als, orig_als0))

# Gate 4: shape (1,28,20) and dtype float32 on augmented output
results["4_shape_dtype"] = bool(
    tuple(x1.shape) == (1, 28, 20) and x1.dtype == torch.float32
)

# extra sanity: original SafeSignal backing tensor not mutated by augmentation
orig_safe0 = StubSubset(SOURCE_SAFESIGNAL).X[0]  # same rng(0) seed -> same values
not_mutated = bool(torch.equal(safe.X[0], orig_safe0))

print("=== AUGMENTATION VERIFICATION GATE ===")
for k, v in results.items():
    print(f"{k}: {'PASS' if v else 'FAIL'}")
print(f"(extra) safesignal_backing_not_mutated: {'PASS' if not_mutated else 'FAIL'}")

all_pass = all(results.values())
print(f"\nALL_GATES: {'PASS' if all_pass else 'FAIL'}")
sys.exit(0 if all_pass else 1)
