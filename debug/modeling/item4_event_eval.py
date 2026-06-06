"""항목4 event-level 평가 + 통계 (read-only). 학습된 15모델 → full 세션 sliding 평가.

precompute(eval_windows.pkl)된 z-SDP 를 15모델(3 policy×5 seed)로 forward 만 해서 세션 단위
event-level 지표를 낸다. 학습/원본 재계산 없음(RPCA 1회 캐싱 재사용).

정의(확정):
  latency_s = (fire_frame_original - onset_original)/100,  onset_original = clean→orig 매핑.
  fire_frame = N연속 positive 확정 window end(original). positive = fall_prob>=thr (+margin).
  timely_3s = fired & 0<=lat<=3 / timely_4s = 0<=lat<=4 / early_fire = fired & lat<0 / late_tp = fired & lat>4.
  forward_recall = forward(stride300) window 만으로 N=1 fire recall.
  event_FAR = non-fall 세션 fire 비율.
threshold 2모드:
  selected_by_val : 모델별 val event sweep(grid) → D-023 rule 로 1 config 선택.
  fixed_baseline  : seed별 fixed 모델 selected config 를 onset 모델에 고정 적용.
단위: non-WALK pooled paired(fixed∩onset_primary) = main / WALK exploratory / subtype 부록.
통계: paired bootstrap CI, McNemar, Wilson CI. 5 seed mean±std. early_fire_rate policy 비교.

제약(read-only): 원본·동결파일·manifest·ckpt 무수정. 산출 item4_results/ (report/json).
"""
from __future__ import annotations
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402
from model.pretrained.model import CNNGRUAttention  # noqa: E402

ITEM4 = ROOT / "debug/modeling/diag_out/onset_detector/item4"
GATE3 = ROOT / "debug/modeling/diag_out/onset_detector/gate3_cache"
FINAL = ROOT / "debug/modeling/diag_out/onset_detector/finalization"
import os as _os  # noqa: E402
CKPT = ROOT / "model/finetune" / _os.environ.get("ITEM4_CKPT_DIR", "checkpoints_item4")
WIN_PKL = ITEM4 / "eval_windows.pkl"
OUTDIR = ITEM4 / "results"
SEEDS = [42, 43, 44, 45, 46]
POLICIES = ["fixed", "onset_primary", "onset_reduced"]
WALK = {"FALL_WALK_F", "FALL_WALK_B"}
WINDOW = 300
THRS = [round(0.10 + 0.05 * i, 2) for i in range(13)]  # 0.10 .. 0.70 (배포 FAR 운영점 탐색)
NS = [1, 2, 3]
MARGINS = [("off", 0.0), ("on_m0.1", 0.1), ("on_m0.2", 0.2), ("on_m0.3", 0.3)]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
import os  # noqa: E402
CKPT_NAME = os.environ.get("ITEM4_CKPT", "best_operating.pt")  # best_val_loss.pt 로 전환 가능


def clean_to_orig(c):
    c = int(c)
    if c < 100:
        return c + 50
    if c < 300:
        return c + 100
    return c + 150


# ── 메타 로드 ────────────────────────────────────────────────────────────────
def load_meta():
    man = {r["filename"]: r for r in csv.DictReader(open(FINAL / "manifest_v2_manual_augmented.csv", encoding="utf-8-sig"))}
    ci = list(csv.DictReader(open(GATE3 / "crop_index.csv", encoding="utf-8-sig")))
    gen = defaultdict(dict)  # filename -> policy -> generated bool
    meta = {}
    for r in ci:
        gen[r["filename"]][r["crop_policy"]] = (r["generated"] == "True")
        meta[r["filename"]] = {"subtype": r["subtype"], "split": r["split_assignment"]}
    # non-fall split from item4 fixed cache
    fx = np.load(ITEM4 / "item4_cache_fixed.npz", allow_pickle=True)
    nf = {}
    for f, y, sp in zip(fx["filename"], fx["y"], fx["split_assignment"]):
        if int(y) != 0:
            nf[str(f)] = {"split": str(sp), "subtype": "nonfall"}
    onset = {fn: (int(float(man[fn]["onset_frame_clean"])) if man[fn]["onset_frame_clean"] not in ("", "None") else None)
             for fn in man}
    return man, meta, gen, nf, onset


# ── 모델 forward → 세션별 window 확률 ────────────────────────────────────────
def load_model(policy, seed):
    p = CKPT / f"{policy}_s{seed}" / CKPT_NAME
    ck = torch.load(p, map_location=DEV, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m = CNNGRUAttention(n_classes=6).to(DEV)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


@torch.no_grad()
def probs_for(model, windows):
    """windows: list of (.., sdp(28,20)). 반환 (n,6) softmax."""
    if not windows:
        return np.empty((0, 6), np.float32)
    X = np.stack([w[-1] for w in windows])[:, None, :, :].astype(np.float32)
    out = []
    for i in range(0, len(X), 512):
        t = torch.from_numpy(X[i:i + 512]).to(DEV)
        out.append(torch.softmax(model(t), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def session_fire(items, prob, thr, N, mval, mmode):
    """items: sweep50 [(s,e,sdp)]. 반환 (fired, fire_end_frame|None)."""
    run = 0
    for k, it in enumerate(items):
        fall = float(prob[k, 0])
        second = float(np.sort(prob[k])[-2])
        ok = fall >= thr and (mmode == "off" or (fall - second) >= mval)
        run = run + 1 if ok else 0
        if run >= N:
            return True, int(it[1])  # window end (original frame)
    return False, None


def forward_fire(fwd_items, prob, thr, mval, mmode):
    for k, it in enumerate(fwd_items):
        if it[0] != "forward":
            continue
        fall = float(prob[k, 0]); second = float(np.sort(prob[k])[-2])
        if fall >= thr and (mmode == "off" or (fall - second) >= mval):
            return True
    return False


# ── config 평가 (세션 집합) ──────────────────────────────────────────────────
def eval_config(prob_cache, fall_fns, nonfall_fns, onset, cfg):
    thr, N, (mmode, mval) = cfg["thr"], cfg["N"], cfg["margin"]
    rec = {"fired": {}, "lat": {}, "fwd_fired": {}, "nf_fired": {}}
    tp = 0
    lat_list = []
    early = timely3 = timely4 = late = fwd_tp = 0
    for fn in fall_fns:
        pc = prob_cache.get(fn)
        if pc is None:
            continue
        fired, fe = session_fire(pc["sweep"], pc["sweep_prob"], thr, N, mval, mmode)
        rec["fired"][fn] = fired
        if fired:
            tp += 1
            on = onset.get(fn)
            if on is not None:
                lat = (fe - clean_to_orig(on)) / 100.0
                rec["lat"][fn] = lat
                lat_list.append(lat)
                if lat < 0:
                    early += 1
                elif lat <= 3.0:
                    timely3 += 1; timely4 += 1
                elif lat <= 4.0:
                    timely4 += 1
                else:
                    late += 1
        ff = forward_fire(pc["forward"], pc["forward_prob"], thr, mval, mmode)
        rec["fwd_fired"][fn] = ff
        if ff:
            fwd_tp += 1
    fp = 0
    for fn in nonfall_fns:
        pcn = prob_cache.get(fn)
        if pcn is None:
            continue
        nfired = session_fire(pcn["sweep"], pcn["sweep_prob"], thr, N, mval, mmode)[0]
        rec["nf_fired"][fn] = nfired
        if nfired:
            fp += 1
    nfall = len([fn for fn in fall_fns if fn in prob_cache])
    nnon = len([fn for fn in nonfall_fns if fn in prob_cache])
    recall = tp / nfall if nfall else float("nan")
    far = fp / nnon if nnon else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    return {
        "recall": recall, "far": far, "f1": f1, "forward_recall": fwd_tp / nfall if nfall else float("nan"),
        "early_fire_rate": early / nfall if nfall else float("nan"),
        "timely_3s": timely3 / nfall if nfall else float("nan"),
        "timely_4s": timely4 / nfall if nfall else float("nan"),
        "late_tp_rate": late / nfall if nfall else float("nan"),
        "latency_median": float(np.median(lat_list)) if lat_list else None,
        "latency_p90": float(np.percentile(lat_list, 90)) if lat_list else None,
        "n_fall": nfall, "n_nonfall": nnon, "tp": tp, "fp": fp,
        "fired": rec["fired"], "lat": rec["lat"],
        "fwd_fired": rec["fwd_fired"], "nf_fired": rec["nf_fired"],
    }


def d023_select(grid_results):
    """grid_results: list of (cfg, metrics on val). D-023 rule."""
    def cands(rmin, fmax):
        return [(c, m) for c, m in grid_results if m["recall"] >= rmin and m["far"] <= fmax]
    for rmin, fmax in ((0.90, 0.10), (0.85, 0.15)):
        cs = cands(rmin, fmax)
        if cs:
            return max(cs, key=lambda cm: cm[1]["f1"])[0]
    # fallback: recall 우선, FAR 초과폭 최소
    return max(grid_results, key=lambda cm: (cm[1]["recall"], -max(0.0, cm[1]["far"] - 0.15)))[0]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - half) / d, (c + half) / d)


def mcnemar(fired_a, fired_b, sessions):
    b = sum(1 for s in sessions if fired_a.get(s) and not fired_b.get(s))
    c = sum(1 for s in sessions if not fired_a.get(s) and fired_b.get(s))
    n = b + c
    # exact binomial p (two-sided) for small n
    from math import comb
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return {"b": b, "c": c, "p": min(1.0, p)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not WIN_PKL.exists():
        print(f"[중단] precompute 없음: {WIN_PKL} (item4_precompute_eval_windows.py 먼저)")
        return 1
    store = pickle.loads(WIN_PKL.read_bytes())
    man, meta, gen, nf, onset = load_meta()

    # 세션 집합
    def split_of(fn):
        if fn in meta:
            return meta[fn]["split"]
        return nf.get(fn, {}).get("split")
    all_sessions = list(store.keys())
    val_nonfall = [s for s in all_sessions if s in nf and nf[s]["split"] == "val"]
    test_nonfall = [s for s in all_sessions if s in nf and nf[s]["split"] == "test"]
    val_fall = [s for s in all_sessions if s in meta and meta[s]["split"] == "val" and s not in nf]
    test_fall = [s for s in all_sessions if s in meta and meta[s]["split"] == "test" and s not in nf]

    def paired(p_a, p_b):
        return [s for s in all_sessions if gen[s].get(p_a) and gen[s].get(p_b)]
    nonwalk = lambda S: [s for s in S if meta.get(s, {}).get("subtype") not in WALK]
    walk = lambda S: [s for s in S if meta.get(s, {}).get("subtype") in WALK]

    paired_fx_on = set(paired("fixed", "onset_primary"))
    test_fall_paired = [s for s in test_fall if s in paired_fx_on]
    val_fall_paired = [s for s in val_fall if s in paired_fx_on]
    print(f"[세션] val fall {len(val_fall)} / test fall {len(test_fall)} | "
          f"nonfall val {len(val_nonfall)} test {len(test_nonfall)}")
    print(f"[paired fixed∩onset_primary] test fall {len(test_fall_paired)} "
          f"(non-WALK {len(nonwalk(test_fall_paired))} / WALK {len(walk(test_fall_paired))})")

    grid = [{"thr": t, "N": n, "margin": m} for t in THRS for n in NS for m in MARGINS]

    # ── 모델별 forward + threshold 선택 + test 평가 ──────────────────────────
    per = {}  # (policy,seed) -> {...}
    for policy in POLICIES:
        for seed in SEEDS:
            key = (policy, seed)
            mp = CKPT / f"{policy}_s{seed}" / CKPT_NAME
            if not mp.exists():
                print(f"[경고] 모델 없음 {key} → skip")
                continue
            model = load_model(policy, seed)
            # 필요한 세션 prob 캐시 (val+test fall+nonfall)
            need = set(val_fall + test_fall + val_nonfall + test_nonfall)
            pcache = {}
            for fn in need:
                st = store.get(fn)
                if st is None:
                    continue
                pcache[fn] = {
                    "sweep": st["sweep50"], "forward": st["forward"],
                    "sweep_prob": probs_for(model, st["sweep50"]),
                    "forward_prob": probs_for(model, st["forward"]),
                }
            del model
            if DEV == "cuda":
                torch.cuda.empty_cache()
            # val grid (D-023 selection) — val 전체 fall/nonfall 기준
            val_grid = [(c, eval_config(pcache, val_fall, val_nonfall, onset, c)) for c in grid]
            sel = d023_select(val_grid)
            light_grid = [(c, {"recall": m["recall"], "far": m["far"], "f1": m["f1"]}) for c, m in val_grid]
            per[key] = {"pcache": pcache, "selected": sel, "val_grid": light_grid}
            print(f"  [{policy} s{seed}] selected thr={sel['thr']} N={sel['N']} margin={sel['margin'][0]}", flush=True)

    # fixed_baseline config: seed별 fixed selected
    fixed_sel = {seed: per[("fixed", seed)]["selected"] for seed in SEEDS if ("fixed", seed) in per}

    # ── test 평가 (2 모드 × policy × seed), 단위별 ───────────────────────────
    def eval_set(policy, seed, cfg, fall_set, nonfall_set):
        pc = per[(policy, seed)]["pcache"]
        return eval_config(pc, fall_set, nonfall_set, onset, cfg)

    UNITS = {
        "nonwalk_paired": nonwalk(test_fall_paired),
        "walk_paired": walk(test_fall_paired),
        "all_test_fall": test_fall,
    }
    results = {"meta": {"seeds": SEEDS, "dev": DEV,
                        "units": {k: len(v) for k, v in UNITS.items()},
                        "nonfall_test": len(test_nonfall),
                        "defs": "timely 0<=lat<=3/4, early<0, late>4; latency time_origin=onset_orig"},
               "selected_configs": {f"{p}_s{s}": per[(p, s)]["selected"] for p in POLICIES for s in SEEDS if (p, s) in per},
               "modes": {}}

    for mode in ("selected_by_val", "fixed_baseline"):
        mode_res = {}
        for unit, fall_set in UNITS.items():
            pol_seed = defaultdict(dict)
            for policy in POLICIES:
                for seed in SEEDS:
                    if (policy, seed) not in per:
                        continue
                    cfg = per[(policy, seed)]["selected"] if mode == "selected_by_val" else fixed_sel.get(seed)
                    if cfg is None:
                        continue
                    m = eval_set(policy, seed, cfg, fall_set, test_nonfall)
                    pol_seed[policy][seed] = m
            # 5-seed mean±std per policy
            agg = {}
            for policy, sd in pol_seed.items():
                ms = list(sd.values())
                keys = ["recall", "far", "f1", "forward_recall", "early_fire_rate",
                        "timely_3s", "timely_4s", "late_tp_rate"]
                agg[policy] = {k: {"mean": float(np.nanmean([m[k] for m in ms])),
                                   "std": float(np.nanstd([m[k] for m in ms]))} for k in keys}
                lat_med = [m["latency_median"] for m in ms if m["latency_median"] is not None]
                agg[policy]["latency_median"] = {"mean": float(np.mean(lat_med)) if lat_med else None}
                agg[policy]["_per_seed"] = sd
            mode_res[unit] = agg
        results["modes"][mode] = mode_res

    # ── 통계: fixed vs onset_primary, non-WALK paired test ───────────────────
    stats = {}
    unit_sessions = nonwalk(test_fall_paired)
    for mode in ("selected_by_val", "fixed_baseline"):
        agg = results["modes"][mode]["nonwalk_paired"]
        if "fixed" not in agg or "onset_primary" not in agg:
            continue
        # McNemar pooled across seeds (fire/no-fire on paired non-WALK test)
        b = c = 0
        for seed in SEEDS:
            if ("fixed", seed) not in per or ("onset_primary", seed) not in per:
                continue
            fx = agg["fixed"]["_per_seed"][seed]["fired"]
            on = agg["onset_primary"]["_per_seed"][seed]["fired"]
            for s in unit_sessions:
                if fx.get(s) and not on.get(s):
                    b += 1
                elif not fx.get(s) and on.get(s):
                    c += 1
        from math import comb
        n = b + c
        if n:
            k = min(b, c)
            pmc = min(1.0, sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2)
        else:
            pmc = 1.0
        # paired bootstrap CI — fall metrics: resample 27 paired fall 세션 (mean over seeds)
        rng = np.random.default_rng(42)
        fall_keys = ["recall", "early_fire_rate", "timely_4s", "forward_recall"]
        diffs = {k: [] for k in fall_keys}
        sess = unit_sessions
        nonfall = test_nonfall
        far_diffs = []
        for _ in range(2000):
            ss = [sess[i] for i in rng.choice(len(sess), len(sess), replace=True)] if sess else []
            for k in fall_keys:
                fx_v, on_v = [], []
                for seed in SEEDS:
                    if ("fixed", seed) not in per:
                        continue
                    fx_v.append(_rate_on(agg["fixed"]["_per_seed"][seed], k, ss))
                    on_v.append(_rate_on(agg["onset_primary"]["_per_seed"][seed], k, ss))
                diffs[k].append(np.nanmean(on_v) - np.nanmean(fx_v))
            # FAR: resample non-fall 세션 (별도 모분)
            ns = [nonfall[i] for i in rng.choice(len(nonfall), len(nonfall), replace=True)] if nonfall else []
            fx_far, on_far = [], []
            for seed in SEEDS:
                if ("fixed", seed) not in per:
                    continue
                fxnf = agg["fixed"]["_per_seed"][seed]["nf_fired"]
                onnf = agg["onset_primary"]["_per_seed"][seed]["nf_fired"]
                fx_far.append(np.mean([1 if fxnf.get(s) else 0 for s in ns]) if ns else np.nan)
                on_far.append(np.mean([1 if onnf.get(s) else 0 for s in ns]) if ns else np.nan)
            far_diffs.append(np.nanmean(on_far) - np.nanmean(fx_far))
        ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.mean(v))] for k, v in diffs.items()}
        ci["far"] = [float(np.percentile(far_diffs, 2.5)), float(np.percentile(far_diffs, 97.5)), float(np.mean(far_diffs))]
        stats[mode] = {"mcnemar": {"b_fixed_only": b, "c_onset_only": c, "p": pmc},
                       "bootstrap_ci_onset_minus_fixed": ci, "n_sessions": len(sess), "n_nonfall": len(nonfall)}
    results["stats_fixed_vs_onset_primary_nonwalk_paired"] = stats

    # ── val recall–FAR frontier (배포 운영점 달성 가능성 진단) ──────────────
    frontier = {}
    for policy in POLICIES:
        grids = [per[(policy, s)]["val_grid"] for s in SEEDS if (policy, s) in per]
        if not grids:
            continue
        pooled = []
        for ci, cfg in enumerate(grid):
            rec = float(np.mean([g[ci][1]["recall"] for g in grids]))
            far = float(np.mean([g[ci][1]["far"] for g in grids]))
            f1 = float(np.mean([g[ci][1]["f1"] for g in grids]))
            pooled.append({"cfg": f"thr{cfg['thr']}_N{cfg['N']}_{cfg['margin'][0]}", "recall": rec, "far": far, "f1": f1})
        caps = {}
        for cap in (0.10, 0.15, 0.20, 0.30, 0.40):
            cs = [p for p in pooled if p["far"] <= cap]
            best = max(cs, key=lambda p: p["recall"]) if cs else None
            caps[f"FAR<={cap}"] = best
        frontier[policy] = {"best_recall_at_far_cap": caps,
                            "max_f1_config": max(pooled, key=lambda p: p["f1"])}
    results["val_frontier"] = frontier

    (OUTDIR / "item4_eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n===== VAL recall–FAR frontier (배포 가능성 진단) =====")
    for policy in POLICIES:
        if policy not in frontier:
            continue
        print(f"\n[{policy}] FAR 상한별 최대 recall (val, 5-seed mean):")
        for cap, best in frontier[policy]["best_recall_at_far_cap"].items():
            if best:
                print(f"  {cap}: recall={best['recall']:.3f} (FAR={best['far']:.3f}, F1={best['f1']:.3f}, {best['cfg']})")
            else:
                print(f"  {cap}: 달성 config 없음")
        mf = frontier[policy]["max_f1_config"]
        print(f"  max-F1: {mf['f1']:.3f} (recall={mf['recall']:.3f} FAR={mf['far']:.3f} {mf['cfg']})")
    print(f"\n[생성] {OUTDIR/'item4_eval_results.json'}")
    _write_report(results, stats, UNITS, test_nonfall)
    return 0


def _rate_on(m, key, sessions):
    """세션 부분집합에서 metric 재계산 (bootstrap용). fired/lat 캐시 사용."""
    fired = m["fired"]; lat = m["lat"]
    nf = len(sessions)
    if nf == 0:
        return float("nan")
    if key == "recall":
        return sum(1 for s in sessions if fired.get(s)) / nf
    if key == "early_fire_rate":
        return sum(1 for s in sessions if fired.get(s) and lat.get(s, 0) < 0) / nf
    if key == "timely_4s":
        return sum(1 for s in sessions if fired.get(s) and 0 <= lat.get(s, -9) <= 4.0) / nf
    if key == "forward_recall":
        fwd = m["fwd_fired"]
        return sum(1 for s in sessions if fwd.get(s)) / nf
    return float("nan")


def _write_report(results, stats, UNITS, test_nonfall):
    L = ["# 항목4 event-level 평가 결과 (fixed vs onset_primary alignment)\n",
         f"단위 N: {results['meta']['units']} | non-fall test {len(test_nonfall)} | seeds {results['meta']['seeds']}",
         f"정의: {results['meta']['defs']}\n",
         "> 경계선 N(non-WALK paired test ~27) — 점추정 단정 금지, CI 중심. WALK/subtype exploratory."]
    for mode in ("selected_by_val", "fixed_baseline"):
        L.append(f"\n## [{mode}] non-WALK paired (main)")
        agg = results["modes"][mode].get("nonwalk_paired", {})
        L.append("policy | recall | FAR | F1 | fwd_recall | early_fire | timely_4s | late_tp")
        L.append("---|---|---|---|---|---|---|---")
        for p in POLICIES:
            if p not in agg:
                continue
            a = agg[p]
            def cell(k):
                return f"{a[k]['mean']:.3f}±{a[k]['std']:.3f}"
            L.append(f"{p} | {cell('recall')} | {cell('far')} | {cell('f1')} | {cell('forward_recall')} | "
                     f"{cell('early_fire_rate')} | {cell('timely_4s')} | {cell('late_tp_rate')}")
        st = stats.get(mode)
        if st:
            ci = st["bootstrap_ci_onset_minus_fixed"]
            L.append(f"\n**통계 (onset_primary − fixed, non-WALK paired N={st['n_sessions']})**")
            L.append(f"- McNemar: fixed-only fire {st['mcnemar']['b_fixed_only']}, onset-only {st['mcnemar']['c_onset_only']}, p={st['mcnemar']['p']:.3f}")
            for k in ["recall", "early_fire_rate", "timely_4s", "far", "forward_recall"]:
                lo, hi, mn = ci[k]
                verdict = "유의(CI 0 미포함)" if (lo > 0 or hi < 0) else "유의차 미검출(CI 0 포함, under-powered 가능)"
                L.append(f"- Δ{k}: {mn:+.3f} [95%CI {lo:+.3f}, {hi:+.3f}] → {verdict}")
    # WALK exploratory early_fire
    L.append("\n## WALK exploratory — early_fire_rate (정렬이 전조 오발화 줄이는지)")
    for mode in ("selected_by_val",):
        agg = results["modes"][mode].get("walk_paired", {})
        for p in POLICIES:
            if p in agg:
                a = agg[p]["early_fire_rate"]
                L.append(f"- [{mode}] {p}: early_fire_rate {a['mean']:.3f}±{a['std']:.3f}")
    fr = results.get("val_frontier", {})
    if fr:
        L.append("\n## ★ val recall–FAR frontier (배포 운영점별 달성 가능 최대 recall, 5-seed mean)")
        L.append("policy | FAR≤0.15 | FAR≤0.20 | FAR≤0.30 | max-F1")
        L.append("---|---|---|---|---")
        for p in POLICIES:
            if p not in fr:
                continue
            cap = fr[p]["best_recall_at_far_cap"]; mf = fr[p]["max_f1_config"]
            def c(k):
                b = cap.get(k)
                return f"R{b['recall']:.3f}/FAR{b['far']:.3f}" if b else "none"
            L.append(f"{p} | {c('FAR<=0.15')} | {c('FAR<=0.2')} | {c('FAR<=0.3')} | "
                     f"F1{mf['f1']:.3f}(R{mf['recall']:.3f}/FAR{mf['far']:.3f})")
        L.append("\n> 배포 운영점은 frontier에서 선택. D-023 selected 표는 목표(FAR≤0.15) 미달 시 max-recall fallback이라 FAR 높음.")
    L.append("\n## Gate3 발견 반영")
    L.append("- onset_primary OOB 88 > onset_reduced 30 (늦은 onset, onset+250>clean400 끝)")
    L.append("- usable_for_onset_aligned 264 중 clean onset 263 (1건 beep구간 수동 onset null)")
    (OUTDIR / "item4_eval_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[생성] {OUTDIR/'item4_eval_report.md'}")
    print("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
