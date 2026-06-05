"""onset_manifest finalization v1: auto_reviewed clean subset + priority review prep (read-only).

게이트 2 detector 기준 확정 후속. final onset_manifest 를 한 번에 완성하지 않는다. 목표:
  (1) manifest_v1_auto_reviewed: 자동 통과분(auto_reviewed)만 학습 가능한 clean subset.
      항목 4 crop alignment 의 provisional input.
  (2) priority high pending_manual 검수 준비물(plot/top-k/reason/추천) — 진규 수동 판정용.

원칙(강제):
  Claude Code 는 pending_manual 의 onset 을 최종 확정하지 않는다. 추천만 한다.
  진규 검수 전까지 pending_manual 은 학습/crop 에 사용하지 않는다.

base detector(확정): nominal[190:350] / k_mad=3.0 / sustain=5 / smooth=5.
soft threshold set A(train+val 확정):
  rise_strength<1.136, rise_slope<0.016, confidence_ref<0.522,
  baseline_noise_ratio>1.206, baseline_mad>0.079, param_sensitivity>51, topk_spread>10.
  topk_spread_extreme = topk_spread>45 (보조 로그). soft_warning_count>=2 → needs_review.

분류:
  excluded            : resampled_count<550
  auto_reviewed       : not needs_review and valid rise
  pending_manual      : needs_review(hard or soft>=2)
    exclude_candidate : rise_not_found and peak_over_baseline<1.10 (no_clear_transient) — 자동제외 확정 X

입력: debug/modeling/diag_out/onset_detector/onset_probe_manifest_long.csv (v0, base row만 사용).
energy 재계산: pending_manual 세션만(peak_over_baseline/top-k/plot). 동결 파일 무수정.
산출: debug/modeling/diag_out/onset_detector/finalization/.
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
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from model.preprocessing.loader import load_safesignal_csv          # noqa: E402
from model.preprocessing.resample import resample_to_100hz          # noqa: E402
from model.preprocessing.rpca import rpca_sparse, DEFAULT_MAX_ITER  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as _e:  # pragma: no cover
    HAVE_MPL = False
    _MPL_ERR = repr(_e)

OUT = ROOT / "debug/modeling/diag_out/onset_detector"
FINAL = OUT / "finalization"
CLEANED = ROOT / "data/cleaned"
MANIFEST_V0 = OUT / "onset_probe_manifest_long.csv"

FIX = {"search_mode": "nominal", "k_mad": "3.0", "sustain_frames": "5"}
BASELINE = (50, 150)
SEARCH = (190, 350)
SMOOTH = 5
K_FIX = 3.0
SUSTAIN_FIX = 5
NOMINAL_ONSET_CLEAN = 100
FALL_ORIG = (200, 400)
BEEP_REGIONS = [(0, 50), (150, 200), (400, 450)]
RECO_NOTE = "Recommendation only. Final onset decision must be approved by Jinkyu."

# set A soft threshold
SOFT_LOW = {"rise_strength": 1.136, "rise_slope": 0.016, "confidence_ref": 0.522}
SOFT_HIGH = {"baseline_noise_ratio": 1.206, "baseline_mad_ratio": 0.079, "param_sensitivity_range": 51.0}
TOPK_ABS = 10.0
TOPK_EXTREME = 45.0
PEAK_OVER_BASELINE_FLOOR = 1.10  # no_clear_transient 보수 기준
WALK_SUBTYPES = {"FALL_WALK_B", "FALL_WALK_F"}


def fnum(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def to_clean(orig):
    if orig is None:
        return None
    if 50 <= orig < 150:
        return orig - 50
    if 200 <= orig < 400:
        return orig - 100
    if 450 <= orig < 550:
        return orig - 150
    return None


def in_beep(orig):
    return any(a <= orig < b for a, b in BEEP_REGIONS)


def soft_flags(base):
    flags = []
    for k, t in SOFT_LOW.items():
        v = fnum(base[k])
        if v is not None and v < t:
            flags.append(f"{k}_low")
    for k, t in SOFT_HIGH.items():
        v = fnum(base[k])
        if v is not None and v > t:
            flags.append({"baseline_noise_ratio": "baseline_noise_high", "baseline_mad_ratio": "baseline_mad_high",
                          "param_sensitivity_range": "param_sensitivity_high"}[k])
    ts = fnum(base["topk_spread"])
    if ts is not None and ts > TOPK_ABS:
        flags.append("topk_spread_high")
    return flags


# ─── energy 재계산 (pending_manual 세션) ─────────────────────────────────────
def recompute_energy(fname):
    path = next(CLEANED.rglob(fname), None)
    if path is None:
        return None
    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    if int(res.resampled_count) < 550:
        return None
    A = res.amplitude[:550]
    e = np.abs(rpca_sparse(A, max_iter=DEFAULT_MAX_ITER, tol=None)).mean(axis=1)
    h = SMOOTH // 2
    es = np.array([e[max(0, i - h):min(len(e), i + h + 1)].mean() for i in range(len(e))])
    return es


def candidates_topk(es, baseline_median, baseline_mad, k=3):
    thr = baseline_median + K_FIX * baseline_mad
    s0, s1 = SEARCH
    cands = []
    for i in range(s0, min(s1, len(es))):
        j = i + SUSTAIN_FIX
        if j <= len(es) and np.all(es[i:j] > thr):
            cands.append(i)
    cands_sorted = sorted(cands, key=lambda i: es[i], reverse=True)[:k]
    out = [{"frame_original": int(i), "frame_clean": to_clean(i), "energy": float(es[i]),
            "in_fall_window": bool(FALL_ORIG[0] <= i < FALL_ORIG[1]), "in_beep": bool(in_beep(i))}
           for i in cands_sorted]
    return out, len(cands), thr


# ─── 분류 ────────────────────────────────────────────────────────────────────
def classify(base, excluded):
    """base detector row → (source, usable, exclude_reason, soft_flags, soft_count, needs_review)."""
    if excluded:
        return "excluded", False, "resampled_lt_550", [], 0, True
    hard = bool(base["needs_review_hard_reasons"])
    flags = soft_flags(base)
    sc = len(flags)
    needs_review = hard or sc >= 2
    valid = base["rise_frame_clean"] != ""
    if not needs_review and valid:
        return "auto_reviewed", True, "", flags, sc, False
    return "pending_manual", False, "", flags, sc, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-energy", type=int, default=None, help="(smoke) energy 재계산 세션 수 제한")
    args = ap.parse_args()
    FINAL.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST_V0, encoding="utf-8-sig")))
    by_file = {}
    for r in rows:
        by_file.setdefault(r["filename"], []).append(r)

    # ── 1차 분류 (manifest only) ─────────────────────────────────────────────
    sessions = []
    for fn, rs in by_file.items():
        excluded = any(r["excluded"] == "True" for r in rs)
        base = next((r for r in rs if r["search_mode"] == FIX["search_mode"]
                     and r["k_mad"] == FIX["k_mad"] and r["sustain_frames"] == FIX["sustain_frames"]), None)
        meta = rs[0]
        if base is None:  # excluded(<550) — base row 없음
            base = rs[0]
        source, usable, excl_reason, flags, sc, nr = classify(base, excluded)
        rnf = "rise_not_found" in (base["needs_review_hard_reasons"] or "")
        sessions.append({
            "fn": fn, "meta": meta, "base": base, "excluded": excluded,
            "source": source, "usable": usable, "exclude_reason": excl_reason,
            "soft_flags": flags, "soft_count": sc, "needs_review": nr, "rise_not_found": rnf,
            "exclude_candidate": False, "exclude_candidate_reason": "",
            "peak_over_baseline": None, "topk": [], "topk_count": 0,
        })

    # ── energy 재계산 (pending_manual 만) → peak_over_baseline / top-k ────────
    pend = [s for s in sessions if s["source"] == "pending_manual"]
    if args.limit_energy:
        pend = pend[:args.limit_energy]
    print(f"[energy] pending_manual {len([s for s in sessions if s['source']=='pending_manual'])}개 중 {len(pend)}개 재계산")
    energy_cache = {}
    for i, s in enumerate(pend):
        bm = fnum(s["base"]["baseline_median"]); bmad = fnum(s["base"]["baseline_mad"])
        es = recompute_energy(s["fn"])
        if es is None or bm is None:
            continue
        energy_cache[s["fn"]] = es
        s["peak_over_baseline"] = float(es[SEARCH[0]:SEARCH[1]].max() / (bm + 1e-9))
        topk, ncand, _ = candidates_topk(es, bm, bmad)
        s["topk"] = topk
        s["topk_count"] = ncand
        # auto_exclude_candidate: rise_not_found and 솟지 않음
        if s["rise_not_found"] and s["peak_over_baseline"] < PEAK_OVER_BASELINE_FLOOR and ncand == 0:
            s["exclude_candidate"] = True
            s["exclude_candidate_reason"] = "no_clear_transient"
        if (i + 1) % 30 == 0 or i + 1 == len(pend):
            print(f"  [{i+1}/{len(pend)}]")

    # ── 추천안 (final 아님) ──────────────────────────────────────────────────
    def recommend(s):
        b = s["base"]
        if s["exclude_candidate"]:
            return f"exclude candidate (no_clear_transient): search max {s['peak_over_baseline']:.2f}x baseline, 0 sustained candidate. {RECO_NOTE}"
        if s["rise_not_found"]:
            if s["peak_over_baseline"] is not None and s["peak_over_baseline"] >= PEAK_OVER_BASELINE_FLOOR:
                return f"weak transient (search max {s['peak_over_baseline']:.2f}x baseline, no sustained-5). manual confirm onset. {RECO_NOTE}"
            return f"no sustained crossing; manual confirm. {RECO_NOTE}"
        hard = b["needs_review_hard_reasons"]
        rise_o = fnum(b["rise_frame_original"]); rise_c = b["rise_frame_clean"]
        in_win = [c for c in s["topk"] if c["in_fall_window"] and not c["in_beep"]]
        if hard and ("beep" in hard or "too_early" in hard):
            if in_win:
                c = in_win[0]
                return f"auto rise(frame {rise_o}) in beep/early; strongest in-window candidate frame {c['frame_original']}(clean {c['frame_clean']}). manual confirm. {RECO_NOTE}"
            return f"auto rise(frame {rise_o}) in beep/early; no clean in-window candidate → likely no_clear_transient. manual review. {RECO_NOTE}"
        # soft-only (valid rise, weak)
        if rise_c != "":
            return f"auto rise frame {rise_o}(clean {rise_c}) but weak (soft: {','.join(s['soft_flags'])}). likely OK, manual confirm. {RECO_NOTE}"
        return f"manual confirm. {RECO_NOTE}"

    for s in sessions:
        s["recommend"] = recommend(s) if s["source"] == "pending_manual" else ""

    # ── manifest_v1 행 구성 ──────────────────────────────────────────────────
    def v1_row(s):
        b, m = s["base"], s["meta"]
        auto = s["source"] == "auto_reviewed"
        rp = b.get("review_priority", "") if not s["excluded"] else "high"
        return {
            "manifest_id": "onset_manifest_v1_2026-06-05", "manifest_version": "v1_auto_reviewed",
            "filename": s["fn"], "env": m["env"], "subject": m["subject"], "activity": m["activity"],
            "subtype": m["subtype"], "trial": m["trial"],
            "split_id": m["split_id"], "split_assignment": m["split_assignment"], "split_unit": "filename",
            "test_sealed": True,
            "source": s["source"], "usable_for_training": auto,
            "exclude_reason": s["exclude_reason"],
            "exclude_candidate": s["exclude_candidate"], "exclude_candidate_reason": s["exclude_candidate_reason"],
            "onset_frame_original": b["rise_frame_original"] if auto else "",
            "onset_frame_clean": b["rise_frame_clean"] if auto else "",
            "rise_frame_original": b["rise_frame_original"] if not s["excluded"] else "",
            "rise_frame_clean": b["rise_frame_clean"] if not s["excluded"] else "",
            "peak_frame_original": b["peak_frame_original"] if not s["excluded"] else "",
            "peak_frame_clean": b["peak_frame_clean"] if not s["excluded"] else "",
            "baseline_median": b["baseline_median"] if not s["excluded"] else "",
            "baseline_mad": b["baseline_mad"] if not s["excluded"] else "",
            "baseline_iqr": b["baseline_iqr"] if not s["excluded"] else "",
            "baseline_p75": b["baseline_p75"] if not s["excluded"] else "",
            "baseline_p90": b["baseline_p90"] if not s["excluded"] else "",
            "baseline_noise_ratio": b["baseline_noise_ratio"] if not s["excluded"] else "",
            "baseline_mad_ratio": b["baseline_mad_ratio"] if not s["excluded"] else "",
            "rise_strength": b["rise_strength"] if not s["excluded"] else "",
            "rise_slope": b["rise_slope"] if not s["excluded"] else "",
            "confidence_ref": b["confidence_ref"] if not s["excluded"] else "",
            "param_sensitivity": b["param_sensitivity_range"] if not s["excluded"] else "",
            "topk_spread": b["topk_spread"] if not s["excluded"] else "",
            "topk_spread_extreme": (fnum(b["topk_spread"]) is not None and fnum(b["topk_spread"]) > TOPK_EXTREME) if not s["excluded"] else "",
            "soft_warning_count": s["soft_count"],
            "needs_review": s["needs_review"],
            "needs_review_hard_reasons": b["needs_review_hard_reasons"] if not s["excluded"] else "resampled_count<550",
            "needs_review_soft_reasons": ";".join(s["soft_flags"]),
            "review_priority": rp,
            "review_priority_reasons": b.get("review_priority_reasons", "") if not s["excluded"] else "hard_resampled_count_lt_550",
        }

    v1_rows = [v1_row(s) for s in sessions]
    V1_COLS = list(v1_rows[0].keys())
    with open(FINAL / "manifest_v1_auto_reviewed.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(V1_COLS)
        for r in v1_rows:
            w.writerow([r[c] for c in V1_COLS])

    # ── provisional onset 분포 (auto_reviewed usable 만) ─────────────────────
    ar = [s for s in sessions if s["source"] == "auto_reviewed"]
    def cl(s):
        return fnum(s["base"]["rise_frame_clean"])
    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else None
    subtypes = sorted({s["meta"]["subtype"] for s in sessions})
    splits = ["train", "val", "test", "out_of_scope"]
    onset_dist = {
        "auto_reviewed_overall_median": med([cl(s) for s in ar]),
        "auto_reviewed_by_subtype_median": {st: med([cl(s) for s in ar if s["meta"]["subtype"] == st]) for st in subtypes},
        "auto_reviewed_by_split_median": {sp: med([cl(s) for s in ar if s["meta"]["split_assignment"] == sp]) for sp in splits},
        "auto_reviewed_count_by_subtype": {st: sum(1 for s in ar if s["meta"]["subtype"] == st) for st in subtypes},
        "auto_reviewed_count_by_split": {sp: sum(1 for s in ar if s["meta"]["split_assignment"] == sp) for sp in splits},
        "note": "provisional — manual 검수(v2) 후 final onset median 재계산. onset-aligned crop은 세션별 onset 사용.",
    }

    # ── 데이터 감소 / 편향 ───────────────────────────────────────────────────
    n = len(sessions)
    def cnt(pred):
        return sum(1 for s in sessions if pred(s))
    src_cnt = {k: cnt(lambda s, k=k: s["source"] == k) for k in ("auto_reviewed", "excluded", "pending_manual")}
    exc_cand = cnt(lambda s: s["exclude_candidate"])
    usable = cnt(lambda s: s["usable"])
    def by_sub(pred):
        return {st: sum(1 for s in sessions if s["meta"]["subtype"] == st and pred(s)) for st in subtypes}
    def by_split(pred):
        return {sp: sum(1 for s in sessions if s["meta"]["split_assignment"] == sp and pred(s)) for sp in splits}
    reduction = {
        "total": n,
        "overall": {"auto_reviewed": src_cnt["auto_reviewed"], "auto_reviewed_rate": src_cnt["auto_reviewed"] / n,
                    "pending_manual": src_cnt["pending_manual"], "pending_manual_rate": src_cnt["pending_manual"] / n,
                    "excluded": src_cnt["excluded"], "auto_exclude_candidate": exc_cand,
                    "usable_for_training": usable, "usable_rate": usable / n},
        "by_subtype": {
            "auto_reviewed": by_sub(lambda s: s["source"] == "auto_reviewed"),
            "pending_manual": by_sub(lambda s: s["source"] == "pending_manual"),
            "excluded": by_sub(lambda s: s["source"] == "excluded"),
            "usable_for_training": by_sub(lambda s: s["usable"]),
        },
        "by_split": {
            "auto_reviewed": by_split(lambda s: s["source"] == "auto_reviewed"),
            "pending_manual": by_split(lambda s: s["source"] == "pending_manual"),
            "usable_for_training": by_split(lambda s: s["usable"]),
        },
        "pending_manual_by_soft_count": {sc: cnt(lambda s, sc=sc: s["source"] == "pending_manual" and s["soft_count"] == sc)
                                         for sc in range(0, 6)},
        "note_test": "test split 비율은 기록만, 기준 변경 미사용.",
    }

    # ── priority review queue (high pending_manual, train/val 우선 정렬) ─────
    def sort_key(s):
        m = s["meta"]
        split_rank = 0 if m["split_assignment"] in ("train", "val") else 1
        prio_rank = 0 if s["base"].get("review_priority") == "high" else 1
        hard_rank = 0 if s["base"]["needs_review_hard_reasons"] else 1
        bn = -(fnum(s["base"]["baseline_noise_ratio"]) or 0)
        ps = -(fnum(s["base"]["param_sensitivity_range"]) or 0)
        sub_rank = 0 if m["subtype"] == "FALL_WALK_B" else (1 if m["subtype"] == "FALL_WALK_F" else 2)
        return (split_rank, prio_rank, hard_rank, bn, ps, sub_rank)

    queue = sorted([s for s in sessions if s["source"] == "pending_manual" and s["base"].get("review_priority") == "high"],
                   key=sort_key)
    with open(FINAL / "priority_review_queue.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["rank", "filename", "subtype", "split_assignment", "review_priority", "hard_reasons",
                    "soft_reasons", "soft_count", "baseline_noise_ratio", "param_sensitivity", "topk_spread",
                    "peak_over_baseline", "topk_count", "walk_baseline_contamination",
                    "topk_cand_frames_clean", "rise_frame_original", "exclude_candidate", "recommendation"])
        for rk, s in enumerate(queue, 1):
            b, m = s["base"], s["meta"]
            walkcontam = (m["subtype"] in WALK_SUBTYPES and (fnum(b["baseline_noise_ratio"]) or 0) > SOFT_HIGH["baseline_noise_ratio"])
            cands = "|".join(str(c["frame_clean"]) for c in s["topk"])
            w.writerow([rk, s["fn"], m["subtype"], m["split_assignment"], b.get("review_priority", ""),
                        b["needs_review_hard_reasons"], ";".join(s["soft_flags"]), s["soft_count"],
                        b["baseline_noise_ratio"], b["param_sensitivity_range"], b["topk_spread"],
                        round(s["peak_over_baseline"], 3) if s["peak_over_baseline"] is not None else "",
                        s["topk_count"], walkcontam, cands, b["rise_frame_original"],
                        s["exclude_candidate"], s["recommend"]])

    # ── plots: train/val high pending_manual ─────────────────────────────────
    plotted = 0
    if HAVE_MPL:
        pdir = FINAL / "plots_priority"
        pdir.mkdir(parents=True, exist_ok=True)
        for s in queue:
            if s["meta"]["split_assignment"] not in ("train", "val"):
                continue
            es = energy_cache.get(s["fn"])
            if es is None:
                continue
            b = s["base"]; bm = fnum(b["baseline_median"]); bmad = fnum(b["baseline_mad"])
            thr = bm + K_FIX * bmad
            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.plot(np.arange(len(es)), es, lw=0.9, color="C0")
            ax.axvspan(*BASELINE, color="0.9"); ax.axvspan(*SEARCH, color="#e6f2ff", alpha=0.6)
            for a, bb in BEEP_REGIONS:
                ax.axvspan(a, bb, color="#ffe6e6", alpha=0.5)
            ax.axhline(thr, color="C2", ls="--", lw=1, label=f"thr={thr:.3f}")
            ro = fnum(b["rise_frame_original"])
            if ro is not None:
                ax.axvline(ro, color="C3", lw=1.6, label=f"auto rise={int(ro)}")
            for c in s["topk"]:
                ax.axvline(c["frame_original"], color="C1", lw=1.0, ls=":", alpha=0.8)
            ax.axvline(200, color="0.5", lw=0.8, ls="-.")
            ax.set_title(f"[REVIEW] {s['fn']} {s['meta']['subtype']} {s['meta']['split_assignment']} "
                         f"| {b['needs_review_hard_reasons']} soft={s['soft_count']} | {s['recommend'][:60]}", fontsize=7)
            ax.set_xlabel("original frame"); ax.set_ylabel("smoothed sparse energy"); ax.legend(fontsize=6)
            fig.tight_layout(); fig.savefig(pdir / f"review_{Path(s['fn']).stem}.png", dpi=110); plt.close(fig)
            plotted += 1

    # ── summary json/md ──────────────────────────────────────────────────────
    payload = {"meta": {"manifest_version": "v1_auto_reviewed", "base_detector": FIX,
                        "soft_set": "A", "principle": "pending_manual onset은 Claude 미확정 — 진규 판정"},
               "classification": reduction, "provisional_onset": onset_dist,
               "priority_queue": {"high_pending_manual_total": len(queue),
                                  "train_val": sum(1 for s in queue if s["meta"]["split_assignment"] in ("train", "val")),
                                  "walk_b": sum(1 for s in queue if s["meta"]["subtype"] == "FALL_WALK_B"),
                                  "walk_f": sum(1 for s in queue if s["meta"]["subtype"] == "FALL_WALK_F"),
                                  "plots_generated": plotted},
               "auto_exclude_candidates": exc_cand}
    (FINAL / "manifest_v1_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    o = reduction["overall"]
    L = ["# onset_manifest v1 (auto_reviewed clean subset) 요약\n",
         f"total {n} | auto_reviewed {o['auto_reviewed']}({o['auto_reviewed_rate']*100:.1f}%) "
         f"pending_manual {o['pending_manual']}({o['pending_manual_rate']*100:.1f}%) "
         f"excluded {o['excluded']} | usable_for_training {o['usable_for_training']}({o['usable_rate']*100:.1f}%)",
         f"auto_exclude_candidate(no_clear_transient): {exc_cand}\n",
         "## subtype별 auto_reviewed / pending_manual / usable",
         "subtype | auto | pending | usable | usable%"]
    for st in subtypes:
        a = reduction["by_subtype"]["auto_reviewed"][st]; p = reduction["by_subtype"]["pending_manual"][st]
        u = reduction["by_subtype"]["usable_for_training"][st]
        tot = sum(1 for s in sessions if s["meta"]["subtype"] == st)
        L.append(f"{st} | {a} | {p} | {u} | {100*u/tot:.0f}%")
    L.append("\n## split별 usable (test는 기록만, 기준변경 미사용)")
    for sp in splits:
        a = reduction["by_split"]["auto_reviewed"][sp]; u = reduction["by_split"]["usable_for_training"][sp]
        tot = sum(1 for s in sessions if s["meta"]["split_assignment"] == sp)
        if tot:
            L.append(f"- {sp}: usable {u}/{tot} ({100*u/tot:.0f}%)")
    L.append("\n## provisional onset median (auto_reviewed usable only, clean frame)")
    L.append(f"- overall: {onset_dist['auto_reviewed_overall_median']}")
    L.append(f"- by_subtype: {onset_dist['auto_reviewed_by_subtype_median']}")
    L.append(f"- by_split: {onset_dist['auto_reviewed_by_split_median']}")
    L.append(f"- auto_reviewed count by_subtype: {onset_dist['auto_reviewed_count_by_subtype']}")
    L.append("\n## priority review queue (high pending_manual)")
    L.append(f"- total {len(queue)} | train/val {payload['priority_queue']['train_val']} | "
             f"WALK_B {payload['priority_queue']['walk_b']} WALK_F {payload['priority_queue']['walk_f']} | plots {plotted}")
    L.append(f"- 위치: {FINAL}/priority_review_queue.csv, plots_priority/")
    L.append("\n## pending_manual soft_count 분포")
    L.append(f"- {reduction['pending_manual_by_soft_count']}")
    (FINAL / "manifest_v1_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n=== 산출 완료 ===")
    for f in ["manifest_v1_auto_reviewed.csv", "manifest_v1_summary.json", "manifest_v1_summary.md", "priority_review_queue.csv"]:
        print(f"  {FINAL / f}")
    print(f"  {FINAL}/plots_priority/  ({plotted}장)")
    print("\n----- manifest_v1_summary.md -----")
    print((FINAL / "manifest_v1_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
