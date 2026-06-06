"""4-arm 더미 학습 평가 (read-only). A/B/C/D × 5 seed, sealed test 원본만.

A fixed_original / B onset_original / C fixed_augmented / D onset_augmented.
item4_event_eval 함수 재사용. eval_windows.pkl(원본 val+test) + checkpoints_item4_arms.
핵심 대조(non-WALK paired test): B vs D(메인), A vs C, (D-C)-(B-A) 결합효과.
2 모드: selected_by_val(arm별 D-023) / fixed_baseline(A 운영점 전 arm 고정).
통계: 5-seed mean±std, paired bootstrap CI, McNemar. class ratio/effective mass 보고.
"""
from __future__ import annotations
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "debug/modeling"))
os.environ.setdefault("ITEM4_CKPT_DIR", "checkpoints_item4_arms")  # env 우선(통제 ckpt 전환 가능)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import item4_event_eval as E  # noqa: E402  (CKPT=checkpoints_item4_arms 적용됨)

ITEM4 = ROOT / "debug/modeling/diag_out/onset_detector/item4"
OUTDIR = ITEM4 / ("arms_results_ctrl" if "ctrl" in os.environ.get("ITEM4_CKPT_DIR", "") else "arms_results")
SEEDS = E.SEEDS
ARMS = ["fixed", "onset_primary", "fixed_aug", "onset_primary_aug"]
LABEL = {"fixed": "A_fixed_original", "onset_primary": "B_onset_original",
         "fixed_aug": "C_fixed_augmented", "onset_primary_aug": "D_onset_augmented"}
GRID = [{"thr": t, "N": n, "margin": m} for t in E.THRS for n in E.NS for m in E.MARGINS]
NONWALK_KEYS = ["recall", "far", "f1", "forward_recall", "early_fire_rate", "timely_4s", "late_tp_rate"]


def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(np.mean(vals))]


def verdict(lo, hi):
    return "유의(CI 0 미포함)" if (lo > 0 or hi < 0) else "유의차 미검출(CI 0 포함, under-powered 가능)"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    store = pickle.loads((ITEM4 / "eval_windows.pkl").read_bytes())
    man, meta, gen, nf, onset = E.load_meta()
    alls = list(store.keys())
    val_nf = [s for s in alls if s in nf and nf[s]["split"] == "val"]
    test_nf = [s for s in alls if s in nf and nf[s]["split"] == "test"]
    val_fall = [s for s in alls if s in meta and meta[s]["split"] == "val" and s not in nf]
    test_fall = [s for s in alls if s in meta and meta[s]["split"] == "test" and s not in nf]
    paired = set(s for s in alls if gen[s].get("fixed") and gen[s].get("onset_primary"))
    nw = lambda S: [s for s in S if meta.get(s, {}).get("subtype") not in E.WALK]
    walk = lambda S: [s for s in S if meta.get(s, {}).get("subtype") in E.WALK]
    test_paired = [s for s in test_fall if s in paired]
    unit_nw = nw(test_paired)
    print(f"[세션] test fall {len(test_fall)} | non-WALK paired {len(unit_nw)} | WALK paired {len(walk(test_paired))} | nonfall test {len(test_nf)}")

    # ── arm×seed: forward + threshold 선택 ────────────────────────────────────
    per = {}
    need = set(val_fall + test_fall + val_nf + test_nf)
    for arm in ARMS:
        for seed in SEEDS:
            mp = E.CKPT / f"{arm}_s{seed}" / E.CKPT_NAME
            if not mp.exists():
                print(f"[경고] 모델 없음 {arm}_s{seed}")
                continue
            model = E.load_model(arm, seed)
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
            per[(arm, seed)] = {"pcache": pcache, "sel": E.d023_select(vg)}
            print(f"  [{LABEL[arm]} s{seed}] sel thr={per[(arm,seed)]['sel']['thr']} N={per[(arm,seed)]['sel']['N']}", flush=True)

    fixed_sel = {s: per[("fixed", s)]["sel"] for s in SEEDS if ("fixed", s) in per}

    # ── 모드별 arm 집계 + per-seed 보존 ───────────────────────────────────────
    results = {"meta": {"seeds": SEEDS, "unit_nonwalk_paired_test": len(unit_nw),
                        "nonfall_test": len(test_nf), "label": LABEL}, "modes": {}}
    perseed = {}  # mode -> arm -> seed -> eval dict (on unit_nw + test_nf)
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
                lat = [m["latency_median"] for m in ms if m["latency_median"] is not None]
                agg[LABEL[arm]]["latency_median"] = float(np.mean(lat)) if lat else None
        results["modes"][mode] = agg

    # ── 대조: B vs D, A vs C, (D-C)-(B-A) (selected_by_val + fixed_baseline) ──
    def paired_boot(mode, a_arm, b_arm):
        """b - a (non-WALK paired test). fall metrics + far + McNemar."""
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
        # McNemar pooled across seeds (fire/no-fire on unit_nw)
        bb = cc = 0
        for s in seeds:
            for x in unit_nw:
                af, bf = A[s]["fired"].get(x), B[s]["fired"].get(x)
                if af and not bf:
                    bb += 1
                elif bf and not af:
                    cc += 1
        nmc = bb + cc
        p = min(1.0, sum(comb(nmc, i) for i in range(min(bb, cc) + 1)) / (2 ** nmc) * 2) if nmc else 1.0
        return {"diff_ci": out, "mcnemar": {"a_only": bb, "b_only": cc, "p": p}, "n_sessions": len(unit_nw)}

    def interaction(mode):
        """((D-C)-(B-A)) for recall, far — 더미가 onset alignment와 결합해 추가효과?"""
        rng = np.random.default_rng(7)
        ps = perseed[mode]
        seeds = [s for s in SEEDS if all((a, s) in per for a in ARMS)]
        rec, far = [], []
        for _ in range(2000):
            ss = [unit_nw[i] for i in rng.choice(len(unit_nw), len(unit_nw), replace=True)] if unit_nw else []
            def mrec(arm):
                return np.nanmean([E._rate_on(ps[arm][s], "recall", ss) for s in seeds])
            rec.append((mrec("onset_primary_aug") - mrec("fixed_aug")) - (mrec("onset_primary") - mrec("fixed")))
            ns = [test_nf[i] for i in rng.choice(len(test_nf), len(test_nf), replace=True)]
            def mfar(arm):
                return np.nanmean([np.mean([1 if ps[arm][s]["nf_fired"].get(x) else 0 for x in ns]) for s in seeds])
            far.append((mfar("onset_primary_aug") - mfar("fixed_aug")) - (mfar("onset_primary") - mfar("fixed")))
        return {"recall_DC_minus_BA": ci(rec), "far_DC_minus_BA": ci(far)}

    stats = {}
    for mode in ("selected_by_val", "fixed_baseline"):
        stats[mode] = {
            "B_vs_D_onset": paired_boot(mode, "onset_primary", "onset_primary_aug"),
            "A_vs_C_fixed": paired_boot(mode, "fixed", "fixed_aug"),
            "interaction_DC_minus_BA": interaction(mode),
        }
    results["stats"] = stats

    # ── class ratio / effective sampled mass ──────────────────────────────────
    from model.finetune.train import safesignal_class_weight_6class, PRETRAINED6_CLASSES
    cr = {}
    for arm in ARMS:
        cache = ITEM4 / (f"item4_cache_{arm}.npz")
        d = np.load(cache, allow_pickle=True)
        tr = d["split_assignment"] == "train"
        y = d["y"][tr]
        cnt = {PRETRAINED6_CLASSES[i]: int((y == i).sum()) for i in range(6)}
        # effective mass within safesignal (source_ratio 0.60 정규화 전 상대)
        raw = {PRETRAINED6_CLASSES[i]: safesignal_class_weight_6class(i, 1.30) * (y == i).sum() for i in range(6)}
        tot = sum(raw.values())
        eff = {k: round(v / tot * 0.60, 4) for k, v in raw.items()}  # source_ratio 0.60
        cr[LABEL[arm]] = {"train_class_count": cnt, "fall_nonfall_ratio": round(cnt["fall"] / (sum(cnt.values()) - cnt["fall"]), 4),
                          "fall_effective_mass": eff["fall"]}
    results["class_ratio_sampler"] = cr

    (OUTDIR / "arms_eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _report(results, stats, cr)
    return 0


def _report(results, stats, cr):
    L = ["# 항목4 더미 4-arm 평가 (within-subject in-domain augmentation 효과)\n",
         f"non-WALK paired test N={results['meta']['unit_nonwalk_paired_test']} | nonfall test {results['meta']['nonfall_test']} | seeds {results['meta']['seeds']}",
         "> 경계선 N — 점추정 단정 금지, CI 중심. WALK/subtype exploratory. train+더미는 train-only(val/test 봉인)."]
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
        for key, lbl in [("B_vs_D_onset", "B→D (onset: 더미 효과, 메인)"), ("A_vs_C_fixed", "A→C (fixed: 더미 효과)")]:
            s = st[key]; d = s["diff_ci"]
            L.append(f"\n**{lbl}** (Δ=aug−orig, N={s['n_sessions']}): McNemar p={s['mcnemar']['p']:.3f}")
            for k in ["recall", "far", "timely_4s", "early_fire_rate", "forward_recall"]:
                lo, hi, mn = d[k]
                L.append(f"- Δ{k}: {mn:+.3f} [{lo:+.3f},{hi:+.3f}] → {verdict(lo,hi)}")
        it = st["interaction_DC_minus_BA"]
        L.append(f"\n**결합효과 (D−C)−(B−A)**: 더미가 onset alignment와 결합해 추가효과?")
        for k, nm in [("recall_DC_minus_BA", "Δrecall"), ("far_DC_minus_BA", "ΔFAR")]:
            lo, hi, mn = it[k]
            L.append(f"- {nm}: {mn:+.3f} [{lo:+.3f},{hi:+.3f}] → {verdict(lo,hi)}")
    L.append("\n## class ratio / effective sampled mass (train)")
    L.append("arm | train fall | fall:nonfall | fall effective mass(source0.60)")
    L.append("---|---|---|---")
    for arm in ARMS:
        lb = LABEL[arm]; c = cr[lb]
        L.append(f"{lb} | {c['train_class_count']['fall']} | {c['fall_nonfall_ratio']} | {c['fall_effective_mass']}")
    L.append("\n## 결론 범위")
    L.append("- within-subject / in-domain augmentation 효과로 제한. 새 subject/env 일반화 해석 금지.")
    L.append("- 더미는 train non-WALK fall origin 기반, train-only. val/test 원본 봉인.")
    (OUTDIR / "arms_eval_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[생성] {OUTDIR}/arms_eval_report.md, arms_eval_results.json")


if __name__ == "__main__":
    raise SystemExit(main())
