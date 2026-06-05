"""게이트 2 확정 라운드: soft threshold 분위수 비교 + 대표 plot (read-only).

base detector 확정: nominal[190:350] / k_mad=3.0 / sustain=5 / smooth=5 (broad=fallback only).
이번 단계:
  (1) soft threshold 를 분위수 3세트(A 10/90, B 15/85, C 20/80)로 비교해 needs_review 총량을
      검수 가능 범위로 맞춘다. train/val 분포로만 threshold 산출(test 봉인).
  (2) 대표 카테고리(broad-only / WALK_B early / rise_not_found / high baseline_noise)
      2~3개씩 sparse energy curve + rise/peak 표시로 sanity check.

soft 조건 (nominal k3/s5 행 기준):
  low(bottom 분위수): rise_strength_low, rise_slope_low, confidence_ref_low
  high(top 분위수)  : baseline_noise_high, baseline_mad_high, param_sensitivity_high
  topk_spread_high  : 절대 >10 또는 top 분위수
  soft_warning_count >= 2 → soft needs_review (복합 약함)
  needs_review = hard(고정 param 행) OR soft_review.  total = union.

입력: 기존 onset_probe_manifest_long.csv / onset_probe_session_summary.csv (재활용, RPCA 재계산 없음).
대표 plot 만 해당 세션 energy 재계산(소수).

제약: read-only, 동결 파일 미수정, train/val만, debug/modeling/diag_out/onset_detector/ 하위.
"""
from __future__ import annotations

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
CLEANED = ROOT / "data/cleaned"
MANIFEST = OUT / "onset_probe_manifest_long.csv"
SESSION_SUMMARY = OUT / "onset_probe_session_summary.csv"

# base detector 확정 param
FIX = {"search_mode": "nominal", "k_mad": "3.0", "sustain_frames": "5"}
BASELINE = (50, 150)
SEARCH_NOMINAL = (190, 350)
SEARCH_BROAD = (180, 400)
SMOOTH = 5
K_FIX = 3.0
TRAIN_VAL = {"train", "val"}
REVIEW_BUDGET_PCT = 35.0  # 초과 시 더 극단(10/90) 권장

# 분위수 세트: bottom pct (top = 100 - bottom)
QSETS = {"A_10_90": 10.0, "B_15_85": 15.0, "C_20_80": 20.0}
LOW_COMPS = ["rise_strength", "rise_slope", "confidence_ref"]
HIGH_COMPS = ["baseline_noise_ratio", "baseline_mad_ratio", "param_sensitivity_range"]
TOPK_ABS = 10.0


def fnum(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ─── manifest 로드 → 세션별 고정-param 행 ────────────────────────────────────
def load_sessions():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    by_file = {}
    for r in rows:
        by_file.setdefault(r["filename"], []).append(r)

    sessions = {}  # filename -> dict(split, subtype, hard, excluded, comps..., fix_row, broad_row)
    for fn, rs in by_file.items():
        split = rs[0]["split_assignment"]
        subtype = rs[0]["subtype"]
        excluded = any(r["excluded"] == "True" for r in rs)
        fix_row = next((r for r in rs if r["search_mode"] == FIX["search_mode"]
                        and r["k_mad"] == FIX["k_mad"] and r["sustain_frames"] == FIX["sustain_frames"]), None)
        broad_row = next((r for r in rs if r["search_mode"] == "broad"
                          and r["k_mad"] == FIX["k_mad"] and r["sustain_frames"] == FIX["sustain_frames"]), None)
        if excluded or fix_row is None:
            sessions[fn] = {"split": split, "subtype": subtype, "excluded": True, "hard": True,
                            "hard_reasons": "resampled_count<550", "comps": {}, "fix_row": None,
                            "broad_row": broad_row, "rise_clean": None}
            continue
        hard = bool(fix_row["needs_review_hard_reasons"])
        comps = {c: fnum(fix_row[c]) for c in (LOW_COMPS + HIGH_COMPS + ["topk_spread"])}
        sessions[fn] = {"split": split, "subtype": subtype, "excluded": False, "hard": hard,
                        "hard_reasons": fix_row["needs_review_hard_reasons"], "comps": comps,
                        "fix_row": fix_row, "broad_row": broad_row,
                        "rise_clean": fnum(fix_row["rise_frame_clean"])}
    return sessions


# ─── 분위수 threshold (train+val 분포) ───────────────────────────────────────
def compute_thresholds(sessions, bottom_pct):
    top_pct = 100.0 - bottom_pct
    tv = [s for s in sessions.values() if s["split"] in TRAIN_VAL and not s["excluded"]]
    thr = {}
    for c in LOW_COMPS:
        vals = [s["comps"][c] for s in tv if s["comps"].get(c) is not None]
        thr[c] = float(np.percentile(vals, bottom_pct)) if vals else None
    for c in HIGH_COMPS:
        vals = [s["comps"][c] for s in tv if s["comps"].get(c) is not None]
        thr[c] = float(np.percentile(vals, top_pct)) if vals else None
    ts = [s["comps"]["topk_spread"] for s in tv if s["comps"].get("topk_spread") is not None]
    thr["topk_spread"] = float(np.percentile(ts, top_pct)) if ts else None
    return thr


def soft_flags(s, thr):
    c = s["comps"]
    flags = []
    if c.get("rise_strength") is not None and thr["rise_strength"] is not None and c["rise_strength"] < thr["rise_strength"]:
        flags.append("rise_strength_low")
    if c.get("rise_slope") is not None and thr["rise_slope"] is not None and c["rise_slope"] < thr["rise_slope"]:
        flags.append("rise_slope_low")
    if c.get("confidence_ref") is not None and thr["confidence_ref"] is not None and c["confidence_ref"] < thr["confidence_ref"]:
        flags.append("confidence_ref_low")
    if c.get("baseline_noise_ratio") is not None and thr["baseline_noise_ratio"] is not None and c["baseline_noise_ratio"] > thr["baseline_noise_ratio"]:
        flags.append("baseline_noise_high")
    if c.get("baseline_mad_ratio") is not None and thr["baseline_mad_ratio"] is not None and c["baseline_mad_ratio"] > thr["baseline_mad_ratio"]:
        flags.append("baseline_mad_high")
    if c.get("param_sensitivity_range") is not None and thr["param_sensitivity_range"] is not None and c["param_sensitivity_range"] > thr["param_sensitivity_range"]:
        flags.append("param_sensitivity_high")
    ts = c.get("topk_spread")
    if ts is not None and (ts > TOPK_ABS or (thr["topk_spread"] is not None and ts > thr["topk_spread"])):
        flags.append("topk_spread_high")
    return flags


# ─── 분위수 세트별 review 통계 ───────────────────────────────────────────────
def evaluate_set(sessions, bottom_pct):
    thr = compute_thresholds(sessions, bottom_pct)
    tv = {fn: s for fn, s in sessions.items() if s["split"] in TRAIN_VAL}
    n = len(tv)
    per_subtype = {}
    soft_count_hist = {}
    n_hard = n_soft = n_total = 0
    soft_increment = 0
    detail = {}
    for fn, s in tv.items():
        flags = soft_flags(s, thr) if not s["excluded"] else []
        sc = len(flags)
        soft_rev = sc >= 2
        hard = s["hard"]
        total = hard or soft_rev
        n_hard += hard
        n_soft += soft_rev
        n_total += total
        if total and not hard:
            soft_increment += 1
        soft_count_hist[sc] = soft_count_hist.get(sc, 0) + 1
        st = s["subtype"]
        d = per_subtype.setdefault(st, {"n": 0, "review": 0, "hard": 0})
        d["n"] += 1
        d["review"] += total
        d["hard"] += hard
        detail[fn] = {"hard": hard, "soft_count": sc, "soft_flags": flags, "review": total}
    return {
        "bottom_pct": bottom_pct, "thresholds": thr, "n_train_val": n,
        "n_hard": n_hard, "n_soft_review": n_soft, "n_total_review": n_total,
        "total_review_pct": 100.0 * n_total / n if n else None,
        "hard_pct": 100.0 * n_hard / n if n else None,
        "soft_increment_over_hard": soft_increment,
        "soft_count_hist": dict(sorted(soft_count_hist.items())),
        "per_subtype": {st: {"n": d["n"], "review_pct": 100.0 * d["review"] / d["n"],
                             "hard_pct": 100.0 * d["hard"] / d["n"]} for st, d in sorted(per_subtype.items())},
        "_detail": detail,
    }


# ─── 대표 세션 energy 재계산 + plot ──────────────────────────────────────────
def recompute_energy(fname):
    path = next(CLEANED.rglob(fname), None)
    if path is None:
        return None
    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    if int(res.resampled_count) < 550:
        return None
    A = res.amplitude[:550]
    sparse = rpca_sparse(A, max_iter=DEFAULT_MAX_ITER, tol=None)
    e = np.abs(sparse).mean(axis=1)
    h = SMOOTH // 2
    es = np.array([e[max(0, i - h):min(len(e), i + h + 1)].mean() for i in range(len(e))])
    return es


def plot_rep(fname, sess, category, plots_dir):
    if not HAVE_MPL:
        return
    es = recompute_energy(fname)
    if es is None:
        return
    b0, b1 = BASELINE
    bm = float(np.median(es[b0:b1])); bmad = float(np.median(np.abs(es[b0:b1] - bm)))
    thr = bm + K_FIX * bmad
    fr = sess["fix_row"]; br = sess["broad_row"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(np.arange(len(es)), es, lw=0.9, color="C0")
    ax.axvspan(b0, b1, color="0.9", label="baseline[50:150]")
    ax.axvspan(*SEARCH_NOMINAL, color="#e6f2ff", alpha=0.6, label="search_nominal")
    for a, b in [(0, 50), (150, 200), (400, 450)]:
        ax.axvspan(a, b, color="#ffe6e6", alpha=0.5)
    ax.axhline(thr, color="C2", ls="--", lw=1, label=f"thr=med+3·mad={thr:.3f}")
    # rise/peak markers (nominal k3 s5)
    if fr is not None and fr["rise_frame_original"]:
        ro = float(fr["rise_frame_original"]); ax.axvline(ro, color="C3", lw=1.6, label=f"rise(nom)={int(ro)}")
        if fr["peak_frame_original"]:
            ax.axvline(float(fr["peak_frame_original"]), color="C1", lw=1.2, ls=":", label=f"peak={int(float(fr['peak_frame_original']))}")
    elif br is not None and br["rise_frame_original"]:
        ax.axvline(float(br["rise_frame_original"]), color="purple", lw=1.6, label=f"rise(broad)={int(float(br['rise_frame_original']))}")
    ax.axvline(200, color="0.5", lw=0.8, ls="-.", label="nominal onset=200")
    ax.set_title(f"[{category}] {fname}  {sess['subtype']} {sess['split']}  "
                 f"noise={sess['comps'].get('baseline_noise_ratio')}", fontsize=8)
    ax.set_xlabel("original frame"); ax.set_ylabel("smoothed sparse energy")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    fig.tight_layout(); fig.savefig(plots_dir / f"rep_{category}_{Path(fname).stem}.png", dpi=120); plt.close(fig)


def pick_representatives(sessions):
    tv = {fn: s for fn, s in sessions.items() if s["split"] in TRAIN_VAL and not s["excluded"]}
    # broad-only: nominal k3/s5 rise 없음 + broad k3/s5 rise 있음
    broad_only = [(fn, s) for fn, s in tv.items()
                  if (s["fix_row"] and not s["fix_row"]["rise_frame_clean"])
                  and (s["broad_row"] and s["broad_row"]["rise_frame_clean"])]
    # WALK_B early: valid rise + 가장 이른 rise_clean
    walkb = sorted([(fn, s) for fn, s in tv.items()
                    if s["subtype"] == "FALL_WALK_B" and s["rise_clean"] is not None],
                   key=lambda x: x[1]["rise_clean"])
    # rise_not_found: nominal k3/s5 rise_not_found
    rnf = [(fn, s) for fn, s in tv.items()
           if s["fix_row"] and "rise_not_found" in s["fix_row"]["needs_review_hard_reasons"]]
    # high baseline noise
    highnoise = sorted([(fn, s) for fn, s in tv.items() if s["comps"].get("baseline_noise_ratio") is not None],
                       key=lambda x: -x[1]["comps"]["baseline_noise_ratio"])
    return {"broad_only": broad_only[:3], "walkb_early": walkb[:3],
            "rise_not_found": rnf[:3], "high_noise": highnoise[:3]}


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"먼저 gate2 probe 실행 필요: {MANIFEST}")
    sessions = load_sessions()
    tv_n = sum(1 for s in sessions.values() if s["split"] in TRAIN_VAL)
    print(f"[load] sessions={len(sessions)} train+val={tv_n} (test 봉인)")

    results = {name: evaluate_set(sessions, bp) for name, bp in QSETS.items()}

    # 권장 세트: total_review <= 35% 중 가장 느슨(20/80)을, 모두 초과면 가장 극단(10/90)
    rec = None
    for name in ["C_20_80", "B_15_85", "A_10_90"]:
        if results[name]["total_review_pct"] is not None and results[name]["total_review_pct"] <= REVIEW_BUDGET_PCT:
            rec = name
            break
    if rec is None:
        rec = "A_10_90"

    # ── 출력: 비교 CSV ───────────────────────────────────────────────────────
    with open(OUT / "onset_soft_threshold_compare.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["set", "bottom_pct", "n_train_val", "hard_pct", "total_review_pct",
                    "soft_increment_over_hard", "WALK_B_review_pct", "WALK_F_review_pct",
                    "SIT_F_review_pct", "recommended"])
        for name, bp in QSETS.items():
            r = results[name]; ps = r["per_subtype"]
            w.writerow([name, bp, r["n_train_val"], round(r["hard_pct"], 1), round(r["total_review_pct"], 1),
                        r["soft_increment_over_hard"],
                        round(ps.get("FALL_WALK_B", {}).get("review_pct", 0), 1),
                        round(ps.get("FALL_WALK_F", {}).get("review_pct", 0), 1),
                        round(ps.get("FALL_SIT_F", {}).get("review_pct", 0), 1),
                        "YES" if name == rec else ""])

    # ── 대표 plot ────────────────────────────────────────────────────────────
    reps = pick_representatives(sessions)
    if HAVE_MPL:
        plots_dir = OUT / "plots_soft"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for cat, items in reps.items():
            for fn, s in items:
                plot_rep(fn, s, cat, plots_dir)

    # ── summary json/md ──────────────────────────────────────────────────────
    clean_results = {name: {k: v for k, v in r.items() if not k.startswith("_")} for name, r in results.items()}
    payload = {
        "meta": {"base_detector": FIX, "review_budget_pct": REVIEW_BUDGET_PCT,
                 "qsets": QSETS, "split": "train+val only (test sealed)",
                 "soft_low": LOW_COMPS, "soft_high": HIGH_COMPS, "topk_abs": TOPK_ABS},
        "results": clean_results, "recommended_set": rec,
        "representatives": {cat: [fn for fn, _ in items] for cat, items in reps.items()},
    }
    (OUT / "onset_soft_threshold_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# 게이트 2 soft threshold 분위수 비교 (nominal k3/s5 고정, train+val, test 봉인)\n",
         f"train+val 세션: {tv_n}  | review budget: {REVIEW_BUDGET_PCT}%\n",
         "## 분위수 세트별 review",
         "set | bottom/top | hard% | total_review% | soft증분 | WALK_B% | WALK_F% | SIT_F%"]
    for name, bp in QSETS.items():
        r = results[name]; ps = r["per_subtype"]
        L.append(f"{name} | {int(bp)}/{int(100-bp)} | {r['hard_pct']:.1f} | {r['total_review_pct']:.1f} | "
                 f"+{r['soft_increment_over_hard']} | "
                 f"{ps.get('FALL_WALK_B',{}).get('review_pct',0):.1f} | "
                 f"{ps.get('FALL_WALK_F',{}).get('review_pct',0):.1f} | "
                 f"{ps.get('FALL_SIT_F',{}).get('review_pct',0):.1f}")
    L.append(f"\n**권장 세트: {rec}** (total_review ≤ {REVIEW_BUDGET_PCT}% 중 가장 느슨, 모두 초과 시 10/90)")
    L.append("\n## soft_warning_count 분포 (세트별)")
    for name in QSETS:
        L.append(f"- {name}: {results[name]['soft_count_hist']}")
    L.append("\n## subtype별 review% (권장 세트 " + rec + ")")
    for st, d in results[rec]["per_subtype"].items():
        L.append(f"- {st}: review {d['review_pct']:.1f}% (hard {d['hard_pct']:.1f}%, n={d['n']})")
    L.append("\n## 대표 plot 카테고리 (plots_soft/)")
    for cat, items in reps.items():
        L.append(f"- {cat}: {[fn for fn, _ in items]}")
    (OUT / "onset_soft_threshold_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n=== 산출 완료 ===")
    for f in ["onset_soft_threshold_compare.csv", "onset_soft_threshold_summary.json", "onset_soft_threshold_summary.md"]:
        print(f"  {OUT / f}")
    if HAVE_MPL:
        print(f"  {OUT/'plots_soft'}/rep_*.png")
    print("\n----- onset_soft_threshold_summary.md -----")
    print((OUT / "onset_soft_threshold_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
