"""Run A / 앵커 checkpoint threshold-sweep 재평가 (read-only).

학습/캐시생성/train.py 실행 없음. 기존 checkpoint(best_operating.pt)를
within_subject test split에서 고정 threshold grid로 재평가한다.

- split 재현: split_safesignal_within_subject(ds, val_ratio=0.2, test_ratio=0.2, seed=43)
- 평가: predict_with_fall_threshold + compute_metrics (train.py helper 재사용)
- cache/checkpoint는 np.load/torch.load read-only로만 접근.
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
    SafeSignalDataset,
    split_safesignal_within_subject,
    predict_with_fall_threshold,
    PRETRAINED6_CLASSES,
)
from model.pretrained.metrics import compute_metrics  # noqa: E402
from model.pretrained.model import CNNGRUAttention  # noqa: E402

SS_CACHE = ROOT / "model/finetune/cache/track1_ss_global6.npz"
CKPTS = {
    "fw1.0": ROOT / "model/finetune/checkpoints_track1_formal_global_fw10_ep60_s43",
    "fw1.5": ROOT / "model/finetune/checkpoints_track1_formal_global_fw15_s43",
}
GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
REF = 0.01
SUBTYPE_MARKERS = ["FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B",
                   "FALL_WALK_F", "FALL_WALK_B"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_split():
    ds = SafeSignalDataset.from_npz(SS_CACHE)
    _, _, test = split_safesignal_within_subject(ds, val_ratio=0.2, test_ratio=0.2, seed=43)
    idx = list(test.indices)
    X = ds.X[idx]
    y = ds.y[idx].numpy().astype(np.int64)
    subj = np.array([ds.subjects[i] for i in idx])
    fnames = [ds.filenames[i] for i in idx]
    return X, y, subj, fnames


def infer_probs(ckpt_dir, X):
    ck = torch.load(ckpt_dir / "best_operating.pt", map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m = CNNGRUAttention(n_classes=6).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = X[i:i + 256].to(device)
            probs.append(torch.softmax(m(xb), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0), ck


def standing_metrics(y, y_pred):
    nonfall = y != 0
    fp_total = int(((y_pred == 0) & nonfall).sum())
    st_idx = PRETRAINED6_CLASSES.index("standing")
    st_fp = int(((y == st_idx) & (y_pred == 0)).sum())
    st_total = int((y == st_idx).sum())
    share = st_fp / fp_total if fp_total else 0.0
    rate = st_fp / st_total if st_total else 0.0
    return share, rate


def per_class_to_fall(y, y_pred):
    out = {}
    for idx, name in enumerate(PRETRAINED6_CLASSES[1:], start=1):
        tot = int((y == idx).sum())
        to_fall = int(((y == idx) & (y_pred == 0)).sum())
        out[name] = to_fall / tot if tot else None
    return out


def per_subject(y, y_pred, subj):
    out = {}
    for s in sorted(set(subj.tolist())):
        mask = subj == s
        ys, yp = y[mask], y_pred[mask]
        fall = ys == 0
        nonfall = ~fall
        recall = int(((yp == 0) & fall).sum()) / int(fall.sum()) if fall.sum() else None
        far = int(((yp == 0) & nonfall).sum()) / int(nonfall.sum()) if nonfall.sum() else None
        out[f"S{int(s):02d}"] = {"recall": recall, "far": far,
                                 "n_fall": int(fall.sum()), "n_nonfall": int(nonfall.sum())}
    return out


def subtype_recall(y, y_pred, fnames):
    fall_mask = y == 0
    out = {}
    for mk in SUBTYPE_MARKERS:
        sel = np.array([fall_mask[i] and mk in Path(fnames[i]).stem.upper()
                        for i in range(len(y))])
        n = int(sel.sum())
        if n == 0:
            out[mk] = {"recall": None, "n": 0}
        else:
            hit = int((y_pred[sel] == 0).sum())
            out[mk] = {"recall": hit / n, "n": n}
    return out


def metric_row(y, probs, t):
    yp = predict_with_fall_threshold(probs, t)
    m = compute_metrics(y, yp, classes=PRETRAINED6_CLASSES)
    share, rate = standing_metrics(y, yp)
    return {"thr": t, "recall": m.fall_recall, "far": m.far, "f1": m.fall_f1,
            "standing_share": share, "standing_rate": rate,
            "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
            "confusion": m.confusion}


def main():
    X, y, subj, fnames = load_test_split()
    result = {"test_n": int(len(y)),
              "y_dist": {PRETRAINED6_CLASSES[i]: int((y == i).sum()) for i in range(6)},
              "subject_dist": {f"S{int(s):02d}": int((subj == s).sum()) for s in sorted(set(subj.tolist()))},
              "checkpoints": {}}

    for tag, cdir in CKPTS.items():
        probs, ck = infer_probs(cdir, X)
        saved_args = ck.get("args", {})
        saved_thr = ck.get("threshold")
        # --- assert: re-eval at saved threshold vs report json ---
        rep_path = cdir / "within_subject_test_report.json"
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        rep_thr = rep["threshold"]
        chk = metric_row(y, probs, float(rep_thr))
        rm = rep["metrics"]
        assert_ok = (
            abs(chk["recall"] - rm["fall_recall"]) < 1e-9
            and abs(chk["far"] - rm["far"]) < 1e-9
            and abs(chk["f1"] - rm["fall_f1"]) < 1e-9
            and chk["confusion"] == rm["confusion"]
        )
        sweep = [metric_row(y, probs, REF)] + [metric_row(y, probs, t) for t in GRID]
        per_thr_detail = {}
        for t in [0.05, 0.10, 0.15, 0.20]:
            yp = predict_with_fall_threshold(probs, t)
            per_thr_detail[f"{t:.2f}"] = {
                "per_class_to_fall": per_class_to_fall(y, yp),
                "per_subject": per_subject(y, yp, subj),
                "subtype_recall": subtype_recall(y, yp, fnames),
            }
        result["checkpoints"][tag] = {
            "ckpt_dir": cdir.name,
            "saved_args": {k: saved_args.get(k) for k in
                           ["class_policy", "split", "seed", "val_ratio", "test_ratio",
                            "safesignal_cache", "fall_weight", "epochs", "patience"]},
            "saved_threshold": saved_thr,
            "report_threshold": rep_thr,
            "assert_pass": bool(assert_ok),
            "assert_detail": {"reeval": {k: chk[k] for k in ["recall", "far", "f1"]},
                              "report": {"fall_recall": rm["fall_recall"], "far": rm["far"], "fall_f1": rm["fall_f1"]},
                              "confusion_match": chk["confusion"] == rm["confusion"]},
            "sweep": sweep,
            "detail_at": per_thr_detail,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
