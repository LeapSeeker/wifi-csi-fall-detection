"""Derive a pretrained6-policy SafeSignal cache from the existing finetune7 cache.

The original builder (debug/modeling/build_safesignal_cache.py) is absent from
this snapshot, but the pretrained6 cache differs from finetune7 ONLY in label
policy:
  - running (finetune7 index 5) is dropped (pretrained6 has no running),
  - picking (finetune7 index 6) is remapped to index 5,
  - fall/walking/sit_stand/lying/standing (0..4) unchanged.

Because the preprocessing pipeline z-normalises each window independently
(model/preprocessing/pipeline.py: sdp = (sdp - sdp.mean()) / (sdp.std()+1e-6)),
removing running windows does NOT change any other window's X values. So this
derivation is bit-exact to what the real builder would emit for the 6 shared
classes. Read-only on the source cache; writes a new .npz.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

FINETUNE7_CLASSES = ("fall", "walking", "sit_stand", "lying", "standing", "running", "picking")
PRETRAINED6_CLASSES = ("fall", "walking", "sit_stand", "lying", "standing", "picking")
RUNNING_IDX = 5
PICKING_IDX = 6


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive pretrained6 cache from finetune7 cache")
    ap.add_argument(
        "--src",
        default="model/finetune/cache/safesignal_e1234_finetune7.npz",
        help="source finetune7 cache (read-only)",
    )
    ap.add_argument(
        "--out",
        default="model/finetune/cache/safesignal_e1234_pretrained6.npz",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"out exists (use --overwrite): {out_path}")

    data = np.load(args.src, allow_pickle=True)
    keys = list(data.files)

    src_classes = [str(c) for c in data["classes"].tolist()] if "classes" in keys else list(FINETUNE7_CLASSES)
    assert tuple(src_classes) == FINETUNE7_CLASSES, f"unexpected source classes: {src_classes}"

    y = data["y"].astype(np.int64, copy=True)
    keep = y != RUNNING_IDX  # drop running

    new_y = y[keep].copy()
    new_y[new_y == PICKING_IDX] = 5  # picking 6 -> 5
    assert new_y.min() >= 0 and new_y.max() <= 5, f"label out of 0..5: {np.unique(new_y)}"

    # Row-aligned arrays to filter; everything else (scalars/classes) handled below.
    per_row_keys = [k for k in keys if k not in {"classes", "class_policy"} and data[k].shape[:1] == y.shape]

    out: dict[str, np.ndarray] = {}
    for k in per_row_keys:
        if k == "y":
            out[k] = new_y
        else:
            out[k] = data[k][keep]
    out["classes"] = np.array(list(PRETRAINED6_CLASSES), dtype=object)
    out["class_policy"] = np.array("pretrained6")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)

    # Summary (mirrors the original builder's reported fields).
    filenames = [str(f) for f in out["filename"].tolist()] if "filename" in out else []
    files_used = len(set(filenames))
    windows = int(len(new_y))
    counts = Counter(int(v) for v in new_y.tolist())
    class_counts = {PRETRAINED6_CLASSES[i]: counts.get(i, 0) for i in range(len(PRETRAINED6_CLASSES))}
    subjects = sorted({int(s) for s in out["subject"].tolist()}) if "subject" in out else []

    print("=== pretrained6 cache (derived from finetune7) ===")
    print(f"out: {out_path}")
    print(f"files_used: {files_used}")
    print(f"windows: {windows}")
    print(f"class_counts: {class_counts}")
    print(f"subjects: {subjects}")
    print(f"keys: {sorted(out.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
