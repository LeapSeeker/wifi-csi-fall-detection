"""fw1.5 3-seed 폴백 운영점 탐색 — thr 0.10/0.15 고정 (read-only).

[1] split/report 재현 assert: saved threshold 재평가가 report json과 일치하는지.
[2] 세 seed 모두 PASS일 때만 thr 0.10/0.15 고정 평가.
원본 파일은 read-only. 새 산출은 stdout + diag_out/ 신규 파일.
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
SEEDS = {s: ROOT / f"model/finetune/checkpoints_track1_formal_global_fw15_s{s}" for s in (42, 43, 44)}
THRS = [0.10, 0.15]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = SafeSignalDataset.from_npz(SS_CACHE)
ARG_KEYS = ["class_policy", "split", "seed", "val_ratio", "test_ratio", "safesignal_cache"]


def split_test(seed):
    _, _, test = split_safesignal_within_subject(ds, val_ratio=0.2, test_ratio=0.2, seed=seed)
    idx = list(test.indices)
    return (ds.X[idx], ds.y[idx].numpy().astype(np.int64),
            np.array([ds.subjects[i] for i in idx]),
            [ds.filenames[i] for i in idx])


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
    return np.concatenate(out, axis=0), ck


def row(y, probs, subj, fnames, thr):
    yp = predict_with_fall_threshold(probs, thr)
    m = compute_metrics(y, yp, classes=PRETRAINED6_CLASSES)
    st = PRETRAINED6_CLASSES.index("standing")
    st_total = int((y == st).sum())
    st_rate = int(((y == st) & (yp == 0)).sum()) / st_total if st_total else None
    fall = y == 0
    wb = np.array([fall[i] and "FALL_WALK_B" in Path(fnames[i]).stem.upper() for i in range(len(y))])
    wb_n = int(wb.sum())
    wb_r = int((yp[wb] == 0).sum()) / wb_n if wb_n else None
    psub = {}
    for s in sorted(set(subj.tolist())):
        mk = subj == s
        ys, yps = y[mk], yp[mk]
        f = ys == 0
        nf = ~f
        psub[f"S{int(s):02d}"] = {
            "recall": int(((yps == 0) & f).sum()) / int(f.sum()) if f.sum() else None,
            "far": int(((yps == 0) & nf).sum()) / int(nf.sum()) if nf.sum() else None,
        }
    return {"recall": m.fall_recall, "far": m.far, "f1": m.fall_f1,
            "standing_rate": st_rate, "FALL_WALK_B_recall": wb_r, "FALL_WALK_B_n": wb_n,
            "confusion": m.confusion, "per_subject": psub}


def mstd(vals):
    a = np.array([v for v in vals if v is not None], float)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std(ddof=0)), 4),
            "vals": [round(float(x), 4) for x in a]}


result = {"asserts": {}, "saved_args": {}, "by_thr": {f"{t:.2f}": {"per_seed": {}} for t in THRS}}
all_pass = True
cache = {}

for s, cdir in SEEDS.items():
    X, y, subj, fnames = split_test(s)
    probs, ck = infer(cdir, X)
    cache[s] = (X, y, subj, fnames, probs)
    saved_args = ck.get("args", {})
    saved_thr = ck.get("threshold")
    result["saved_args"][s] = {k: saved_args.get(k) for k in ARG_KEYS} | {"saved_threshold": saved_thr}
    rep = json.loads((cdir / "within_subject_test_report.json").read_text(encoding="utf-8"))
    rm = rep["metrics"]
    chk = row(y, probs, subj, fnames, float(rep["threshold"]))
    ok = (abs(chk["recall"] - rm["fall_recall"]) < 1e-9 and abs(chk["far"] - rm["far"]) < 1e-9
          and abs(chk["f1"] - rm["fall_f1"]) < 1e-9 and chk["confusion"] == rm["confusion"])
    # arg checks
    arg_ok = (saved_args.get("class_policy") == "pretrained6"
              and saved_args.get("split") == "within_subject"
              and saved_args.get("seed") == s
              and saved_args.get("val_ratio") == 0.2
              and saved_args.get("test_ratio") == 0.2
              and "track1_ss_global6.npz" in str(saved_args.get("safesignal_cache", "")))
    result["asserts"][s] = {"report_match": bool(ok), "args_ok": bool(arg_ok),
                            "report_thr": rep["threshold"],
                            "reeval": {k: round(chk[k], 6) for k in ("recall", "far", "f1")},
                            "report": {"recall": round(rm["fall_recall"], 6), "far": round(rm["far"], 6), "f1": round(rm["fall_f1"], 6)},
                            "confusion_match": chk["confusion"] == rm["confusion"]}
    all_pass = all_pass and ok and arg_ok

result["all_assert_pass"] = all_pass

if all_pass:
    for t in THRS:
        tk = f"{t:.2f}"
        for s in SEEDS:
            X, y, subj, fnames, probs = cache[s]
            r = row(y, probs, subj, fnames, t)
            r.pop("confusion")
            result["by_thr"][tk]["per_seed"][s] = r
        ps = result["by_thr"][tk]["per_seed"]
        result["by_thr"][tk]["agg"] = {
            "recall": mstd([ps[s]["recall"] for s in SEEDS]),
            "far": mstd([ps[s]["far"] for s in SEEDS]),
            "f1": mstd([ps[s]["f1"] for s in SEEDS]),
            "FALL_WALK_B_recall": mstd([ps[s]["FALL_WALK_B_recall"] for s in SEEDS]),
            "all_far_le_015": all(ps[s]["far"] <= 0.15 for s in SEEDS),
            "far_individual": {s: round(ps[s]["far"], 4) for s in SEEDS},
        }
        result["by_thr"][tk]["per_subject_3seed"] = {
            sub: {"recall_mean": round(float(np.mean([ps[s]["per_subject"][sub]["recall"] for s in SEEDS])), 4),
                  "far_mean": round(float(np.mean([ps[s]["per_subject"][sub]["far"] for s in SEEDS])), 4),
                  "recall_by_seed": {s: round(ps[s]["per_subject"][sub]["recall"], 4) for s in SEEDS},
                  "far_by_seed": {s: round(ps[s]["per_subject"][sub]["far"], 4) for s in SEEDS}}
            for sub in ("S01", "S02", "S03")
        }

print(json.dumps(result, indent=2, ensure_ascii=False))
