"""balanced 증강 4-arm 평가 (read-only). A/B/C/D × 5 seed, sealed test 원본만.

A fixed / B onset_primary / C fixed_balanced_aug / D onset_primary_balanced_aug.
item4_event_eval(E) 함수 재사용 + eval_windows.pkl(원본 val+test). 단일 ckpt root 가정 금지 —
arm별 root: A/B→checkpoints_item4_reg(없으면 balanced root fallback), C/D→checkpoints_item4_balanced.

대조: Primary D−B(onset, balanced 효과) / Secondary C−A(fixed) / Interaction (D−B)−(C−A) 참고.
평가: selected_by_val + fixed_baseline + frontier(FAR<=0.15/0.20/0.30/max-F1, val 선택→test 적용).
통계: per-seed Δ, paired bootstrap 95% CI, 5-seed mean±std, McNemar.
판정: statistically supported(CI 0 미포함) / directional candidate(4/5 동방향 & 임계 & 반대지표 가드) / null.
low_diversity_contribution(nonfall_quality_report) 결론부 자동 반영.

제약(read-only): 원본/동결/manifest 무수정. 신규 리포트만.
"""
from __future__ import annotations
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "debug/modeling"))
os.environ.setdefault("ITEM4_CKPT_DIR", "checkpoints_item4_balanced")  # E.CKPT 기본(미사용, arm root 별도)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import item4_event_eval as E  # noqa: E402

ITEM4 = ROOT / "debug/modeling/diag_out/onset_detector/item4"
REG_ROOT = ROOT / "model/finetune/checkpoints_item4_reg"
BAL_ROOT = ROOT / "model/finetune" / os.environ.get("ITEM4_BAL_CKPT_DIR", "checkpoints_item4_balanced")
OUTDIR = ROOT / "debug/modeling/balanced_aug"
NF_QUALITY = ROOT / "debug/dummy_gen/balanced/nonfall_quality_report.json"
SEEDS = E.SEEDS
ARMS = ["fixed", "onset_primary", "fixed_balanced_aug", "onset_primary_balanced_aug"]
LABEL = {"fixed": "A_fixed", "onset_primary": "B_onset_primary",
         "fixed_balanced_aug": "C_fixed_balanced_aug", "onset_primary_balanced_aug": "D_onset_primary_balanced_aug"}
GRID = [{"thr": t, "N": n, "margin": m} for t in E.THRS for n in E.NS for m in E.MARGINS]
NONWALK_KEYS = ["recall", "far", "f1", "forward_recall", "early_fire_rate", "timely_4s", "late_tp_rate"]
FAR_CAPS = [0.15, 0.20, 0.30]


def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(np.mean(vals))]


def verdict_ci(lo, hi):
    return "유의(CI 0 미포함)" if (lo > 0 or hi < 0) else "유의차 미검출(CI 0 포함)"


AB_ROOT_MODE = "auto"  # CLI --ab-root: reg|balanced|auto


def arm_root(arm, seed):
    """A/B(fixed,onset_primary) ckpt root 결정. C/D 는 항상 balanced.
    auto: balanced 에 재학습본이 있으면 최신으로 우선, 없으면 reg(재사용)."""
    if arm not in ("fixed", "onset_primary"):
        return BAL_ROOT
    if AB_ROOT_MODE == "reg":
        return REG_ROOT
    if AB_ROOT_MODE == "balanced":
        return BAL_ROOT
    bal_ok = (BAL_ROOT / f"{arm}_s{seed}" / E.CKPT_NAME).exists()
    return BAL_ROOT if bal_ok else REG_ROOT


def load_model_root(arm, seed, root):
    import torch
    p = root / f"{arm}_s{seed}" / E.CKPT_NAME
    ck = torch.load(p, map_location=E.DEV, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m = E.CNNGRUAttention(n_classes=6).to(E.DEV)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


def sel_far_cap(vg, cap):
    cs = [(c, m) for c, m in vg if not np.isnan(m["far"]) and m["far"] <= cap]
    if cs:
        return max(cs, key=lambda cm: (cm[1]["recall"], cm[1]["f1"]))[0]
    return min(vg, key=lambda cm: (m if not np.isnan((m := cm[1]["far"])) else 9))[0]


def sel_maxf1(vg):
    return max(vg, key=lambda cm: cm[1]["f1"])[0]


def directional(deltas, metric):
    """deltas: per-seed Δ list. metric in {far,recall}. 스펙 §7 directional candidate."""
    n = len(deltas)
    if n == 0:
        return False, "no_seeds"
    mean = float(np.mean(deltas))
    if metric == "far":
        same = sum(1 for d in deltas if d < 0)        # 개선 = 감소
        ok_dir = same >= 4 and mean <= -0.05
    else:  # recall
        same = sum(1 for d in deltas if d > 0)
        ok_dir = same >= 4 and mean >= 0.03
    return ok_dir, f"{same}/{n} 동방향, mean={mean:+.3f}"


def main():
    global AB_ROOT_MODE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab-root", choices=["reg", "balanced", "auto"], default="auto",
                    help="A/B(fixed,onset_primary) ckpt root. auto=재학습본(balanced) 우선, 없으면 reg")
    ap.add_argument("--require-seeds", type=int, default=0,
                    help="arm별 최소 seed ckpt 수(미만이면 fail-fast). 최종 실험은 5 권장")
    args = ap.parse_args()
    AB_ROOT_MODE = args.ab_root
    print(f"[config] ab_root={AB_ROOT_MODE} require_seeds={args.require_seeds}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not (ITEM4 / "eval_windows.pkl").exists():
        print(f"[FAIL-FAST] eval_windows.pkl 없음 — item4_precompute_eval_windows.py 먼저")
        return 1
    store = pickle.loads((ITEM4 / "eval_windows.pkl").read_bytes())
    man, meta, gen, nf, onset = E.load_meta()
    alls = list(store.keys())
    val_nf = [s for s in alls if s in nf and nf[s]["split"] == "val"]
    test_nf = [s for s in alls if s in nf and nf[s]["split"] == "test"]
    val_fall = [s for s in alls if s in meta and meta[s]["split"] == "val" and s not in nf]
    test_fall = [s for s in alls if s in meta and meta[s]["split"] == "test" and s not in nf]
    paired = set(s for s in alls if gen[s].get("fixed") and gen[s].get("onset_primary"))
    nw = lambda S: [s for s in S if meta.get(s, {}).get("subtype") not in E.WALK]
    test_paired = [s for s in test_fall if s in paired]
    unit_nw = nw(test_paired)
    print(f"[세션] test fall {len(test_fall)} | non-WALK paired {len(unit_nw)} | nonfall test {len(test_nf)}")

    # ── arm×seed forward + 선택 ───────────────────────────────────────────────
    per = {}
    used_root = {}
    need = set(val_fall + test_fall + val_nf + test_nf)
    for arm in ARMS:
        for seed in SEEDS:
            root = arm_root(arm, seed)
            mp = root / f"{arm}_s{seed}" / E.CKPT_NAME
            if not mp.exists():
                print(f"[경고] 모델 없음 {arm}_s{seed} (root={root.name})")
                continue
            used_root[(arm, seed)] = root.name
            model = load_model_root(arm, seed, root)
            pcache = {}
            for fn in need:
                st = store.get(fn)
                if st is None:
                    continue
                pcache[fn] = {"sweep": st["sweep50"], "forward": st["forward"],
                              "sweep_prob": E.probs_for(model, st["sweep50"]),
                              "forward_prob": E.probs_for(model, st["forward"])}
            del model
            import torch
            if E.DEV == "cuda":
                torch.cuda.empty_cache()
            vg = [(c, E.eval_config(pcache, val_fall, val_nf, onset, c)) for c in GRID]
            per[(arm, seed)] = {"pcache": pcache, "vg": vg, "sel": E.d023_select(vg)}
            print(f"  [{LABEL[arm]} s{seed}] root={root.name} sel thr={per[(arm,seed)]['sel']['thr']} "
                  f"N={per[(arm,seed)]['sel']['N']}", flush=True)

    if args.require_seeds:
        for arm in ARMS:
            have = sum(1 for s in SEEDS if (arm, s) in per)
            if have < args.require_seeds:
                print(f"[FAIL-FAST] {LABEL[arm]} seed ckpt {have} < require_seeds {args.require_seeds}")
                return 1
        print(f"[require-seeds] 전 arm >= {args.require_seeds} seed OK")

    fixed_sel = {s: per[("fixed", s)]["sel"] for s in SEEDS if ("fixed", s) in per}

    # ── 모드별 집계 + per-seed ────────────────────────────────────────────────
    results = {"meta": {"seeds": SEEDS, "unit_nonwalk_paired_test": len(unit_nw),
                        "nonfall_test": len(test_nf), "label": LABEL, "used_root": used_root,
                        "primary": "D-B (onset balanced 효과)", "secondary": "C-A (fixed)"},
               "modes": {}}
    perseed = {}
    for mode in ("selected_by_val", "fixed_baseline"):
        agg = {}
        perseed[mode] = {}
        for arm in ARMS:
            sd = {}
            for seed in SEEDS:
                if (arm, seed) not in per:
                    continue
                cfg = per[(arm, seed)]["sel"] if mode == "selected_by_val" else fixed_sel.get(seed)
                if cfg is None:
                    continue
                sd[seed] = E.eval_config(per[(arm, seed)]["pcache"], unit_nw, test_nf, onset, cfg)
            perseed[mode][arm] = sd
            if sd:
                ms = list(sd.values())
                agg[LABEL[arm]] = {k: {"mean": float(np.nanmean([m[k] for m in ms])),
                                       "std": float(np.nanstd([m[k] for m in ms]))} for k in NONWALK_KEYS}
        results["modes"][mode] = agg

    # ── paired bootstrap (D-B primary, C-A secondary, interaction) ────────────
    def paired_boot(mode, a_arm, b_arm):
        rng = np.random.default_rng(42)
        A, B = perseed[mode][a_arm], perseed[mode][b_arm]
        seeds = [s for s in SEEDS if s in A and s in B]
        fall_keys = ["recall", "early_fire_rate", "timely_4s", "forward_recall"]
        diffs = {k: [] for k in fall_keys}
        far = []
        for _ in range(2000):
            ss = [unit_nw[i] for i in rng.choice(len(unit_nw), len(unit_nw), replace=True)] if unit_nw else []
            for k in fall_keys:
                diffs[k].append(np.nanmean([E._rate_on(B[s], k, ss) for s in seeds]) -
                                np.nanmean([E._rate_on(A[s], k, ss) for s in seeds]))
            ns = [test_nf[i] for i in rng.choice(len(test_nf), len(test_nf), replace=True)]
            fa = np.nanmean([np.mean([1 if A[s]["nf_fired"].get(x) else 0 for x in ns]) for s in seeds])
            fb = np.nanmean([np.mean([1 if B[s]["nf_fired"].get(x) else 0 for x in ns]) for s in seeds])
            far.append(fb - fa)
        out = {k: ci(v) for k, v in diffs.items()}
        out["far"] = ci(far)
        return {"diff_ci": out, "n_sessions": len(unit_nw), "seeds": seeds}

    def perseed_delta(mode, a_arm, b_arm):
        """per-seed Δ(b−a): recall(unit_nw), far(test_nf)."""
        A, B = perseed[mode][a_arm], perseed[mode][b_arm]
        seeds = [s for s in SEEDS if s in A and s in B]
        d_rec, d_far = [], []
        for s in seeds:
            d_rec.append(E._rate_on(B[s], "recall", unit_nw) - E._rate_on(A[s], "recall", unit_nw))
            fa = np.mean([1 if A[s]["nf_fired"].get(x) else 0 for x in test_nf]) if test_nf else float("nan")
            fb = np.mean([1 if B[s]["nf_fired"].get(x) else 0 for x in test_nf]) if test_nf else float("nan")
            d_far.append(fb - fa)
        return d_rec, d_far, seeds

    def interaction(mode):
        rng = np.random.default_rng(7)
        ps = perseed[mode]
        seeds = [s for s in SEEDS if all((a, s) in per for a in ARMS)]
        rec, far = [], []
        for _ in range(2000):
            ss = [unit_nw[i] for i in rng.choice(len(unit_nw), len(unit_nw), replace=True)] if unit_nw else []
            mrec = lambda arm: np.nanmean([E._rate_on(ps[arm][s], "recall", ss) for s in seeds])
            rec.append((mrec("onset_primary_balanced_aug") - mrec("onset_primary")) -
                       (mrec("fixed_balanced_aug") - mrec("fixed")))
            ns = [test_nf[i] for i in rng.choice(len(test_nf), len(test_nf), replace=True)]
            mfar = lambda arm: np.nanmean([np.mean([1 if ps[arm][s]["nf_fired"].get(x) else 0 for x in ns]) for s in seeds])
            far.append((mfar("onset_primary_balanced_aug") - mfar("onset_primary")) -
                       (mfar("fixed_balanced_aug") - mfar("fixed")))
        return {"recall_DB_minus_CA": ci(rec), "far_DB_minus_CA": ci(far)}

    stats = {}
    for mode in ("selected_by_val", "fixed_baseline"):
        st = {"primary_D_vs_B": paired_boot(mode, "onset_primary", "onset_primary_balanced_aug"),
              "secondary_C_vs_A": paired_boot(mode, "fixed", "fixed_balanced_aug"),
              "interaction_DB_minus_CA": interaction(mode)}
        # 판정 (primary D-B)
        d_rec, d_far, seeds = perseed_delta(mode, "onset_primary", "onset_primary_balanced_aug")
        far_lo, far_hi, far_mn = st["primary_D_vs_B"]["diff_ci"]["far"]
        rec_lo, rec_hi, rec_mn = st["primary_D_vs_B"]["diff_ci"]["recall"]
        far_supported = far_lo > 0 or far_hi < 0
        rec_supported = rec_lo > 0 or rec_hi < 0
        # 방향 분리: FAR 감소(<0)=개선, recall 증가(>0)=개선
        far_label = None if not far_supported else ("improvement" if far_mn < 0 else "regression")
        rec_label = None if not rec_supported else ("improvement" if rec_mn > 0 else "regression")
        supported_improve = far_label == "improvement" or rec_label == "improvement"
        supported_regress = far_label == "regression" or rec_label == "regression"
        far_dir, far_note = directional(d_far, "far")
        rec_dir, rec_note = directional(d_rec, "recall")
        mean_rec = float(np.mean(d_rec)) if d_rec else float("nan")
        mean_far = float(np.mean(d_far)) if d_far else float("nan")
        far_cand = far_dir and (mean_rec >= -0.05)   # FAR 개선 후보: recall 과도악화 가드
        rec_cand = rec_dir and (mean_far <= 0.05)    # recall 개선 후보: FAR 과도악화 가드
        if supported_improve and supported_regress:
            jud = "statistically_supported_mixed"     # 한쪽 개선·다른쪽 악화 동시 유의
        elif supported_improve:
            jud = "statistically_supported_improvement"
        elif supported_regress:
            jud = "statistically_supported_regression"  # 유의한 악화 — 명시 분리
        elif far_cand or rec_cand:
            jud = "directional_candidate"
        else:
            jud = "null"
        st["judgement_primary_D_vs_B"] = {
            "verdict": jud, "far_ci": [far_lo, far_hi, far_mn], "recall_ci": [rec_lo, rec_hi, rec_mn],
            "far_label": far_label, "recall_label": rec_label,
            "far_supported": far_supported, "recall_supported": rec_supported,
            "far_directional": [far_dir, far_note], "recall_directional": [rec_dir, rec_note],
            "far_candidate_guarded": far_cand, "recall_candidate_guarded": rec_cand,
            "per_seed_delta_recall": [round(x, 4) for x in d_rec],
            "per_seed_delta_far": [round(x, 4) for x in d_far]}
        stats[mode] = st
    results["stats"] = stats

    # ── frontier (val 선택 → test 적용) ───────────────────────────────────────
    def frontier_point(label, selector):
        row = {"point": label}
        arm_means = {}
        for arm in ARMS:
            recs, fars, f1s = [], [], []
            for seed in SEEDS:
                if (arm, seed) not in per:
                    continue
                cfg = selector(per[(arm, seed)]["vg"])
                m = E.eval_config(per[(arm, seed)]["pcache"], unit_nw, test_nf, onset, cfg)
                recs.append(m["recall"]); fars.append(m["far"]); f1s.append(m["f1"])
            if recs:
                arm_means[arm] = {"recall": float(np.nanmean(recs)), "far": float(np.nanmean(fars)),
                                  "f1": float(np.nanmean(f1s)),
                                  "recall_std": float(np.nanstd(recs)), "far_std": float(np.nanstd(fars))}
        row["arms"] = {LABEL[a]: arm_means[a] for a in ARMS if a in arm_means}
        if "onset_primary" in arm_means and "onset_primary_balanced_aug" in arm_means:
            b, d = arm_means["onset_primary"], arm_means["onset_primary_balanced_aug"]
            row["primary_D_minus_B"] = {"recall": d["recall"] - b["recall"], "far": d["far"] - b["far"]}
        if "fixed" in arm_means and "fixed_balanced_aug" in arm_means:
            a, c = arm_means["fixed"], arm_means["fixed_balanced_aug"]
            row["secondary_C_minus_A"] = {"recall": c["recall"] - a["recall"], "far": c["far"] - a["far"]}
        return row

    frontier = []
    for cap in FAR_CAPS:
        frontier.append(frontier_point(f"FAR<={cap}", lambda vg, cap=cap: sel_far_cap(vg, cap)))
    frontier.append(frontier_point("max_F1", sel_maxf1))
    results["frontier"] = frontier

    # ── class ratio / effective mass ──────────────────────────────────────────
    from model.finetune.train import safesignal_class_weight_6class, PRETRAINED6_CLASSES
    cr = {}
    for arm in ARMS:
        cache = ITEM4 / f"item4_cache_{arm}.npz"
        if not cache.exists():
            continue
        d = np.load(cache, allow_pickle=True)
        tr = d["split_assignment"] == "train"
        y = d["y"][tr]
        cnt = {PRETRAINED6_CLASSES[i]: int((y == i).sum()) for i in range(6)}
        cr[LABEL[arm]] = {"train_class_count": cnt,
                          "fall_nonfall_ratio": round(cnt["fall"] / max(1, sum(cnt.values()) - cnt["fall"]), 4)}
    results["class_ratio"] = cr

    # ── low diversity (nonfall quality) ───────────────────────────────────────
    low_div_classes = []
    if NF_QUALITY.exists():
        try:
            low_div_classes = json.loads(NF_QUALITY.read_text(encoding="utf-8")).get("low_diversity_classes", [])
        except Exception:
            pass
    results["low_diversity_classes"] = low_div_classes

    (OUTDIR / "balanced_eval_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_frontier_csv(frontier)
    _report(results, stats, cr, frontier, low_div_classes)
    return 0


def _write_frontier_csv(frontier):
    rows = []
    for fp in frontier:
        for lab, m in fp.get("arms", {}).items():
            rows.append({"point": fp["point"], "arm": lab, "recall": round(m["recall"], 4),
                         "far": round(m["far"], 4), "f1": round(m["f1"], 4),
                         "recall_std": round(m["recall_std"], 4), "far_std": round(m["far_std"], 4)})
    with open(OUTDIR / "balanced_eval_frontier.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["point", "arm", "recall", "far", "f1", "recall_std", "far_std"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (OUTDIR / "balanced_eval_frontier.json").write_text(
        json.dumps(frontier, indent=2, ensure_ascii=False), encoding="utf-8")


def _report(results, stats, cr, frontier, low_div_classes):
    L = ["# balanced 증강 4-arm 평가 (fall+non-fall 더미, class-ratio 통제)\n",
         f"non-WALK paired test N={results['meta']['unit_nonwalk_paired_test']} | "
         f"nonfall test {results['meta']['nonfall_test']} | seeds {results['meta']['seeds']}",
         f"Primary: {results['meta']['primary']} | Secondary: {results['meta']['secondary']}",
         "> 경계선 N — CI 중심 해석. 더미 train-only, val/test 원본 봉인. test 튜닝 없음(val 선택)."]
    for mode in ("selected_by_val", "fixed_baseline"):
        L.append(f"\n## [{mode}] non-WALK paired test (5-seed mean±std)")
        L.append("arm | recall | FAR | F1 | fwd_recall | early_fire | timely_4s | late_tp")
        L.append("---|---|---|---|---|---|---|---")
        agg = results["modes"][mode]
        for arm in ARMS:
            lb = LABEL[arm]
            if lb not in agg:
                continue
            a = agg[lb]
            L.append(f"{lb} | " + " | ".join(f"{a[k]['mean']:.3f}±{a[k]['std']:.3f}" for k in
                     ["recall", "far", "f1", "forward_recall", "early_fire_rate", "timely_4s", "late_tp_rate"]))
        st = stats[mode]
        for key, lbl in [("primary_D_vs_B", "Primary D−B (onset balanced)"),
                         ("secondary_C_vs_A", "Secondary C−A (fixed balanced)")]:
            s = st[key]; d = s["diff_ci"]
            L.append(f"\n**{lbl}** (Δ=balanced−orig, N={s['n_sessions']}):")
            for k in ["recall", "far", "timely_4s", "early_fire_rate"]:
                lo, hi, mn = d[k]
                L.append(f"- Δ{k}: {mn:+.3f} [{lo:+.3f},{hi:+.3f}] → {verdict_ci(lo,hi)}")
        j = st["judgement_primary_D_vs_B"]
        L.append(f"\n**판정 (Primary D−B): `{j['verdict']}`** "
                 f"(FAR sig={j['far_label']}, recall sig={j['recall_label']}; 개선=FAR↓/recall↑)")
        L.append(f"- FAR Δ per-seed {j['per_seed_delta_far']} | directional {j['far_directional']} | guarded_cand={j['far_candidate_guarded']}")
        L.append(f"- recall Δ per-seed {j['per_seed_delta_recall']} | directional {j['recall_directional']} | guarded_cand={j['recall_candidate_guarded']}")
        it = st["interaction_DB_minus_CA"]
        for k, nm in [("recall_DB_minus_CA", "Δrecall"), ("far_DB_minus_CA", "ΔFAR")]:
            lo, hi, mn = it[k]
            L.append(f"- interaction (D−B)−(C−A) {nm}: {mn:+.3f} [{lo:+.3f},{hi:+.3f}] (참고)")
    L.append("\n## frontier (val 선택 → test, 5-seed mean)")
    L.append("point | arm | recall | FAR | F1")
    L.append("---|---|---|---|---")
    for fp in frontier:
        for lab, m in fp.get("arms", {}).items():
            L.append(f"{fp['point']} | {lab} | {m['recall']:.3f} | {m['far']:.3f} | {m['f1']:.3f}")
        if "primary_D_minus_B" in fp:
            p = fp["primary_D_minus_B"]
            L.append(f"  → D−B: Δrecall {p['recall']:+.3f}, ΔFAR {p['far']:+.3f}")
    L.append("\n## class ratio (train)")
    L.append("arm | train fall | fall:nonfall")
    L.append("---|---|---")
    for arm in ARMS:
        lb = LABEL[arm]
        if lb in cr:
            c = cr[lb]
            L.append(f"{lb} | {c['train_class_count']['fall']} | {c['fall_nonfall_ratio']}")
    L.append("\n## 결론 / 해석 가드")
    L.append("- within-subject / in-domain augmentation 효과로 제한. 새 subject/env 일반화 해석 금지.")
    L.append("- 더미는 train origin 기반 train-only. val/test 원본 봉인.")
    if low_div_classes:
        L.append(f"- ⚠ low_diversity_contribution=true: {low_div_classes} — 해당 class dummy use 수 부족으로 "
                 f"다양성 기여 제한적. balanced 전체 결과 해석 시 이 한계를 반영.")
    else:
        L.append("- low_diversity_contribution: 해당 class 없음.")
    (OUTDIR / "balanced_eval_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[생성] {OUTDIR}/balanced_eval_report.md, balanced_eval_results.json, balanced_eval_frontier.csv|json")


if __name__ == "__main__":
    raise SystemExit(main())
