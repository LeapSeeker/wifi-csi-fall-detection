"""게이트 1 마무리: WALK_B 좌표계 session-level recall 이득 정량화 (read-only).

WALK_B 세션은 전부 실제 fall. 따라서 crop 별 (fall_prob >= 0.1) 세션 비율 =
그 crop 의 WALK_B session-level fall recall. clean400_concat 유지가 WALK_B recall 에
주는 실제 이득을 기록한다. (좌표계 재판정 아님 — smoothing probe 로 splice artifact
우려는 이미 해소. 본 보강은 recall 근거 정량화.)

crop 후보:
  concat_main          = clean[50:350] = orig[100:150]+[200:400]+[450:500]   (주)
  continuous_center    = orig[150:450]                                       (주)
  continuous_pre_fall  = orig[100:400]                                       (보조)
  continuous_fall_post = orig[200:500]                                       (보조)

경로: raw crop(300,n_sc) → window_to_model_input(=RPCA→ACF→SDP→z) → (1,28,20)
      → (1,1,28,20) → finetuned_baseline6 → softmax.

대상: 자체수집 FALL_WALK_B 전체, resampled_count >= 550. 미달은 skip 기록.

제약: read-only, 동결 파일/데이터/학습 무수정. 기존 산출물 덮어쓰지 않음(새 파일명).
      산출: debug/modeling/diag_out/beep_concat_artifact/ 하위.

사용: python debug/modeling/gate1b_walkb_detection_rate.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from model.preprocessing.loader import load_safesignal_csv, parse_safesignal_filename  # noqa: E402
from model.preprocessing.resample import resample_to_100hz                              # noqa: E402

from gate1b_walkb_occlusion import (                                                     # noqa: E402
    load_model, infer, preprocess_crop, CKPT, CKPT_ID, PRETRAINED6_CLASSES, FALL_IDX,
)
import torch  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as _e:  # pragma: no cover
    HAVE_MPL = False
    _MPL_ERR = repr(_e)

OUT = ROOT / "debug/modeling/diag_out/beep_concat_artifact"
CLEANED = ROOT / "data/cleaned"
SUBTYPE = "FALL_WALK_B"
MIN_FRAMES = 550
THR = 0.1

CANDIDATES = ["concat_main", "continuous_center", "continuous_pre_fall", "continuous_fall_post"]
MAIN_PAIR = ("concat_main", "continuous_center")


def build_all_crops(A550):
    clean = np.concatenate([A550[50:150], A550[200:400], A550[450:550]], axis=0)
    return {
        "concat_main": clean[50:350],
        "continuous_center": A550[150:450],
        "continuous_pre_fall": A550[100:400],
        "continuous_fall_post": A550[200:500],
    }


def infer_fields(crop, model, device):
    pp = preprocess_crop(crop)
    p, _ = infer(model, pp["model_input"], device, want_attention=False)
    fall = float(p[FALL_IDX])
    pred_idx = int(p.argmax())
    return {
        "fall_prob": fall,
        "crosses_thr_0p1": bool(fall >= THR),
        "pred_class": PRETRAINED6_CLASSES[pred_idx],
        "pred_prob": float(p[pred_idx]),
        "fall_rank": int((p > p[FALL_IDX]).sum()) + 1,
        "max_nonfall_prob": float(p[1:].max()),
        "fall_margin": float(fall - p[1:].max()),
    }


def stats(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return {k: None for k in ("min", "p25", "median", "p75", "max")}
    return {"min": float(a.min()), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="(smoke) 처리 세션 수 제한")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    if not CKPT.exists():
        raise FileNotFoundError(f"checkpoint 없음: {CKPT}")
    model, ck, classes = load_model(device)
    print(f"[ckpt] {CKPT_ID}: classes={classes} threshold={ck.get('threshold')} (THR={THR})")

    files = sorted(CLEANED.rglob(f"*{SUBTYPE}*_T*.csv"))
    print(f"[scan] {SUBTYPE} 후보 파일 {len(files)}개")

    sessions_meta = []          # 메타/포함여부
    per_cand_rows = []          # 세션 × candidate
    included = []               # 분석된 세션 dict
    n_done = 0
    for f in files:
        try:
            m = parse_safesignal_filename(f)
            env, subj, trial = m.environment, m.subject, m.trial
        except Exception as e:
            sessions_meta.append(dict(file=f.name, env="", subject="", trial="", subtype=SUBTYPE,
                                      resampled_count="", selection_group="",
                                      included=False, skip_reason=f"parse_error:{repr(e)[:60]}"))
            continue
        sel_group = "primary_e1_s01" if (env == 1 and subj == 1) else "walk_cross_subject"
        try:
            raw = load_safesignal_csv(f, rx="both")
            res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
            n = int(res.resampled_count)
        except Exception as e:
            sessions_meta.append(dict(file=f.name, env=env, subject=subj, trial=trial, subtype=SUBTYPE,
                                      resampled_count="", selection_group=sel_group,
                                      included=False, skip_reason=f"load_error:{repr(e)[:60]}"))
            continue
        if n < MIN_FRAMES:
            sessions_meta.append(dict(file=f.name, env=env, subject=subj, trial=trial, subtype=SUBTYPE,
                                      resampled_count=n, selection_group=sel_group,
                                      included=False, skip_reason=f"resampled_count<{MIN_FRAMES}"))
            continue

        A550 = res.amplitude[:550]
        crops = build_all_crops(A550)
        fields = {c: infer_fields(crops[c], model, device) for c in CANDIDATES}
        sessions_meta.append(dict(file=f.name, env=env, subject=subj, trial=trial, subtype=SUBTYPE,
                                  resampled_count=n, selection_group=sel_group,
                                  included=True, skip_reason=""))
        rec = {"file": f.name, "env": env, "subject": subj, "trial": trial,
               "selection_group": sel_group, "fields": fields}
        included.append(rec)
        for c in CANDIDATES:
            per_cand_rows.append({"file": f.name, "env": env, "subject": subj, "trial": trial,
                                  "candidate": c, **fields[c]})
        cm = fields["concat_main"]; ct = fields["continuous_center"]
        print(f"[an] E{env}_S{subj:02d}_T{trial:03d} n={n} | concat={cm['fall_prob']:.3f}"
              f"({'O' if cm['crosses_thr_0p1'] else 'x'}) center={ct['fall_prob']:.3f}"
              f"({'O' if ct['crosses_thr_0p1'] else 'x'})")
        n_done += 1
        if args.limit and n_done >= args.limit:
            print(f"[limit] {args.limit}세션에서 중단(smoke)")
            break

    # ── 집계 ──────────────────────────────────────────────────────────────────
    n_inc = len(included)
    by_cand = {}
    for c in CANDIDATES:
        fps = [r["fields"][c]["fall_prob"] for r in included]
        det = sum(1 for r in included if r["fields"][c]["crosses_thr_0p1"])
        amax = sum(1 for r in included if r["fields"][c]["pred_class"] == "fall")
        st = stats(fps)
        by_cand[c] = {
            "n_sessions": n_inc, "detected_count_thr0p1": det,
            "session_recall_thr0p1": (det / n_inc) if n_inc else None,
            "fall_prob_min": st["min"], "fall_prob_p25": st["p25"], "fall_prob_median": st["median"],
            "fall_prob_p75": st["p75"], "fall_prob_max": st["max"],
            "argmax_fall_count": amax, "argmax_fall_rate": (amax / n_inc) if n_inc else None,
        }

    # concat vs center 4분할
    both = concat_only = center_only = neither = 0
    for r in included:
        cd = r["fields"]["concat_main"]["crosses_thr_0p1"]
        td = r["fields"]["continuous_center"]["crosses_thr_0p1"]
        both += cd and td
        concat_only += cd and not td
        center_only += td and not cd
        neither += (not cd) and (not td)
    recall_gain = None
    if n_inc:
        recall_gain = by_cand["concat_main"]["session_recall_thr0p1"] - by_cand["continuous_center"]["session_recall_thr0p1"]
    quad = {"both_detected_count": both, "concat_only_detected_count": concat_only,
            "center_only_detected_count": center_only, "neither_detected_count": neither,
            "recall_gain_concat_minus_center": recall_gain}

    # subject/env 분해
    by_es = {}
    for r in included:
        key = f"E{r['env']}_S{r['subject']:02d}"
        d = by_es.setdefault(key, {"n_sessions": 0, "concat_detected": 0, "center_detected": 0,
                                   "concat_only": 0, "center_only": 0})
        cd = r["fields"]["concat_main"]["crosses_thr_0p1"]
        td = r["fields"]["continuous_center"]["crosses_thr_0p1"]
        d["n_sessions"] += 1
        d["concat_detected"] += int(cd); d["center_detected"] += int(td)
        d["concat_only"] += int(cd and not td); d["center_only"] += int(td and not cd)

    # ── 산출물 ────────────────────────────────────────────────────────────────
    with open(OUT / "walkb_sessions.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        cols = ["file", "env", "subject", "trial", "subtype", "resampled_count",
                "selection_group", "included", "skip_reason"]
        w.writerow(cols)
        for s in sessions_meta:
            w.writerow([s.get(k, "") for k in cols])

    with open(OUT / "walkb_detection_rate_by_candidate.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        cols = ["file", "env", "subject", "trial", "candidate", "fall_prob", "crosses_thr_0p1",
                "pred_class", "pred_prob", "fall_rank", "max_nonfall_prob", "fall_margin"]
        w.writerow(cols)
        for r in per_cand_rows:
            w.writerow([r["file"], r["env"], r["subject"], r["trial"], r["candidate"],
                        round(r["fall_prob"], 5), r["crosses_thr_0p1"], r["pred_class"],
                        round(r["pred_prob"], 5), r["fall_rank"], round(r["max_nonfall_prob"], 5),
                        round(r["fall_margin"], 5)])

    summary = {
        "meta": {"checkpoint_id": CKPT_ID, "threshold": THR, "subtype": SUBTYPE,
                 "n_candidate_files": len(files), "n_included": n_inc,
                 "n_skipped": len(sessions_meta) - n_inc,
                 "note": "WALK_B 전부 실제 fall → crosses_thr0.1 비율 = session-level fall recall."},
        "by_candidate": by_cand,
        "concat_vs_center_quadrant": quad,
        "by_env_subject": by_es,
    }
    (OUT / "walkb_detection_rate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # md
    L = ["# WALK_B 좌표계 session-level recall (clean400_concat 이득 정량화)\n",
         f"checkpoint: {CKPT_ID} | THR={THR} | WALK_B 전부 실제 fall → detection rate = recall",
         f"후보 파일 {len(files)} / included {n_inc} / skipped {len(sessions_meta)-n_inc}\n",
         "## candidate별 session-level recall",
         "candidate | n | detected | recall | fall_prob med(p25~p75) | argmax_fall_rate"]
    for c in CANDIDATES:
        d = by_cand[c]
        rec = "n/a" if d["session_recall_thr0p1"] is None else f"{d['session_recall_thr0p1']:.3f}"
        med = "n/a" if d["fall_prob_median"] is None else f"{d['fall_prob_median']:.3f}"
        p25 = "n/a" if d["fall_prob_p25"] is None else f"{d['fall_prob_p25']:.3f}"
        p75 = "n/a" if d["fall_prob_p75"] is None else f"{d['fall_prob_p75']:.3f}"
        ar = "n/a" if d["argmax_fall_rate"] is None else f"{d['argmax_fall_rate']:.3f}"
        L.append(f"{c} | {d['n_sessions']} | {d['detected_count_thr0p1']} | {rec} | {med}({p25}~{p75}) | {ar}")
    L.append("\n## concat_main vs continuous_center")
    L.append(f"- concat recall = {by_cand['concat_main']['session_recall_thr0p1']}")
    L.append(f"- center recall = {by_cand['continuous_center']['session_recall_thr0p1']}")
    L.append(f"- **recall_gain (concat - center) = {recall_gain}**")
    L.append(f"- both={quad['both_detected_count']} concat_only={quad['concat_only_detected_count']} "
             f"center_only={quad['center_only_detected_count']} neither={quad['neither_detected_count']}")
    L.append("\n## subject/env 분해 (concat_det / center_det / concat_only / center_only / n)")
    for key in sorted(by_es):
        d = by_es[key]
        L.append(f"- {key}: {d['concat_detected']} / {d['center_detected']} / "
                 f"{d['concat_only']} / {d['center_only']} / {d['n_sessions']}")
    share = "\n".join(L) + "\n"
    (OUT / "walkb_detection_rate_summary.md").write_text(share, encoding="utf-8")

    # plots
    if HAVE_MPL and n_inc:
        plots_dir = OUT / "plots_detection_rate"
        plots_dir.mkdir(parents=True, exist_ok=True)
        # bar: detection rate by candidate
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        rates = [by_cand[c]["session_recall_thr0p1"] for c in CANDIDATES]
        ax.bar(range(len(CANDIDATES)), rates, color=["C0", "C1", "C2", "C3"])
        for i, v in enumerate(rates):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(CANDIDATES))); ax.set_xticklabels(CANDIDATES, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("session recall (fall_prob>=0.1)")
        ax.set_title(f"WALK_B session-level recall by crop (n={n_inc})", fontsize=9)
        fig.tight_layout(); fig.savefig(plots_dir / "walkb_recall_by_candidate.png", dpi=120); plt.close(fig)
        # scatter: concat vs center fall_prob
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        cx = [r["fields"]["concat_main"]["fall_prob"] for r in included]
        cy = [r["fields"]["continuous_center"]["fall_prob"] for r in included]
        ax.scatter(cx, cy, s=22, alpha=0.7)
        ax.axhline(THR, color="C3", ls="--", lw=0.8); ax.axvline(THR, color="C3", ls="--", lw=0.8)
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.7)
        ax.set_xlabel("concat_main fall_prob"); ax.set_ylabel("continuous_center fall_prob")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_title("WALK_B: concat vs center (낙상 세션)", fontsize=9)
        fig.tight_layout(); fig.savefig(plots_dir / "walkb_concat_vs_center_scatter.png", dpi=120); plt.close(fig)

    print("\n=== 산출 완료 ===")
    for f in ["walkb_sessions.csv", "walkb_detection_rate_by_candidate.csv",
              "walkb_detection_rate_summary.json", "walkb_detection_rate_summary.md"]:
        print(f"  {OUT / f}")
    print("\n----- walkb_detection_rate_summary.md -----")
    print(share)


if __name__ == "__main__":
    main()
