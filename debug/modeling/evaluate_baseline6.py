"""Alsaify baseline best.pt를 SafeSignal 6-class cache에서 평가한다."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.pretrained.metrics import compute_metrics, format_report, save_metrics
from model.pretrained.model import CLASSES, CNNGRUAttention


def _load_state(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def evaluate(args: argparse.Namespace) -> None:
    cache = np.load(args.cache, allow_pickle=True)
    X = cache["X"].astype(np.float32, copy=False)
    y = cache["y"].astype(np.int64, copy=False)
    classes = [str(x) for x in cache["classes"].tolist()] if "classes" in cache else list(CLASSES)
    if tuple(classes) != tuple(CLASSES):
        raise SystemExit(
            "FAIL: baseline6 평가는 pretrained6 cache가 필요합니다. "
            f"cache classes={classes}, expected={list(CLASSES)}"
        )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CNNGRUAttention(n_classes=len(CLASSES)).to(device)
    model.load_state_dict(_load_state(args.ckpt, device), strict=True)
    model.eval()

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    probs: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(axis=1))

    y_prob = np.concatenate(probs, axis=0)
    y_pred = np.concatenate(preds, axis=0).astype(np.int64)
    metrics = compute_metrics(y, y_pred, classes=CLASSES)
    print(format_report(metrics, header="SafeSignal baseline6 evaluation"))

    out = Path(args.out) if args.out else Path(args.cache).with_suffix(".baseline6_metrics.json")
    out = (PROJECT_ROOT / out).resolve() if not out.is_absolute() else out
    save_metrics(metrics, out)
    extra = {
        "cache": str(Path(args.cache).resolve()),
        "ckpt": str(Path(args.ckpt).resolve()),
        "device": str(device),
        "n_samples": int(len(y)),
        "mean_fall_probability": float(y_prob[:, 0].mean()) if len(y_prob) else 0.0,
    }
    extra_path = out.with_suffix(".run.json")
    extra_path.write_text(json.dumps({**asdict(metrics), "run": extra}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] {out}")
    print(f"[write] {extra_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained 6-class best.pt on SafeSignal cache")
    parser.add_argument("--cache", type=Path, required=True, help="pretrained6 policy cache")
    parser.add_argument("--ckpt", type=Path, default=PROJECT_ROOT / "model" / "pretrained" / "checkpoints" / "best.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out", default=None, help="metrics JSON 경로")
    return parser.parse_args()


def main() -> int:
    evaluate(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
