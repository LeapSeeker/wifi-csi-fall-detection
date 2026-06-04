"""fw1.5 운영점 3-seed 재현 평가 — threshold 0.05 고정 (read-only).

각 seed는 자기 split(seed로 split_safesignal_within_subject 재현)의 sealed test에서
해당 seed의 best_operating.pt를 thr=0.05로 평가. selector threshold는 무시.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model.finetune.train import (  # noqa: E402
    SafeSignalDataset, split_safesignal_within_subject,
    predict_with_fall_threshold, PRETRAINED6_CLASSES,
)
from model.pretrained.metrics import compute_metrics  # noqa: E402
from model.pretrained.model import CNNGRUAttention  # noqa: E402

SS_CACHE = ROOT / "model/finetune/cache/track1_ss_global6.npz"
THR = 0.05
SEEDS = {
    42: ROOT / "model/finetune/checkpoints_track1_formal_global_fw15_s42",
    43: ROOT / "model/finetune/checkpoints_track1_formal_global_fw15_s43",
    44: ROOT / "model/finetune/checkpoints_track1_formal_global_fw15_s44",
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = SafeSignalDataset.from_npz(SS_CACHE)


def split_test(seed):
    _, _, test = split_safesignal_within_subject(ds, val_ratio=0.2, test_ratio=0.2, seed=seed)
    idx = list(test.indices)
    X = ds.X[idx]
    y = ds.y[idx].numpy().astype(np.int64)
    subj = np.array([ds.subjects[i] for i in idx])
    fnames = [ds.filenames[i] for i in idx]
    return X, y, subj, fnames


def infer(ckpt_dir, X):
    ck = torch.load(ckpt_dir / "best_operating.pt", map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m = CNNGRUAttention(n_classes=6).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(torch.softmax(m(X[i:i + 256].to(device)), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate(seed, ckpt_dir):
    X, y, subj, fnames = split_test(seed)
    probs = infer(ckpt_dir, X)
    yp = predict_with_fall_threshold(probs, THR)
    m = compute_metrics(y, yp, classes=PRETRAINED6_CLASSES)
    st = PRETRAINED6_CLASSES.index("standing")
    st_total = int((y == st).sum())
    st_rate = int(((y == st) & (yp == 0)).sum()) / st_total if st_total else None
    # FALL_WALK_B recall
    fall = y == 0
    wb = np.array([fall[i] and "FALL_WALK_B" in Path(fnames[i]).stem.upper() for i in range(len(y))])
    wb_n = int(wb.sum())
    wb_recall = int((yp[wb] == 0).sum()) / wb_n if wb_n else None
    # per-subject
    psub = {}
    for s in sorted(set(subj.tolist())):
        mask = subj == s
        ys, yps = y[mask], yp[mask]
        f = ys == 0
        nf = ~f
        psub[f"S{int(s):02d}"] = {
            "recall": int(((yps == 0) & f).sum()) / int(f.sum()) if f.sum() else None,
            "far": int(((yps == 0) & nf).sum()) / int(nf.sum()) if nf.sum() else None,
        }
    return {
        "seed": seed, "test_n": int(len(y)),
        "recall": m.fall_recall, "far": m.far, "f1": m.fall_f1,
        "standing_rate": st_rate,
        "FALL_WALK_B_recall": wb_recall, "FALL_WALK_B_n": wb_n,
        "per_subject": psub,
        "fall_n": int(fall.sum()),
    }


def stats(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)), "vals": [round(float(x), 4) for x in a]}


res = {str(s): evaluate(s, d) for s, d in SEEDS.items()}
agg = {
    "recall": stats([res[str(s)]["recall"] for s in SEEDS]),
    "far": stats([res[str(s)]["far"] for s in SEEDS]),
    "f1": stats([res[str(s)]["f1"] for s in SEEDS]),
    "FALL_WALK_B_recall": stats([res[str(s)]["FALL_WALK_B_recall"] for s in SEEDS]),
    "standing_rate": stats([res[str(s)]["standing_rate"] for s in SEEDS]),
}
persub = {}
for sub in ["S01", "S02", "S03"]:
    persub[sub] = {
        "recall_mean": float(np.mean([res[str(s)]["per_subject"][sub]["recall"] for s in SEEDS])),
        "far_mean": float(np.mean([res[str(s)]["per_subject"][sub]["far"] for s in SEEDS])),
        "recall_by_seed": {str(s): round(res[str(s)]["per_subject"][sub]["recall"], 4) for s in SEEDS},
        "far_by_seed": {str(s): round(res[str(s)]["per_subject"][sub]["far"], 4) for s in SEEDS},
    }
print(json.dumps({"per_seed": res, "aggregate": agg, "per_subject_3seed": persub}, indent=2, ensure_ascii=False))
