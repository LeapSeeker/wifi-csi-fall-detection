"""트랙1 D-020 FORMAL ablation — A(both global) vs B(both per-lag) 비교 분석.

BANNER: D-020 FORMAL ablation — SafeSignal+Alsaify 둘 다 동일 정규화,
        A=both global / B=both per-lag.

within_subject_test_report.json (sealed test, operating threshold) 만 사용.
주 기준 = paired A vs B delta (같은 seed). A control vs baseline 0.781/0.134 는 reference-only
(Alsaify 행순서가 baseline 과 달라 재현이 더 어긋날 수 있음 → gate-stop 안 함).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FT = PROJECT_ROOT / "model" / "finetune"
DIAG = PROJECT_ROOT / "debug" / "modeling" / "diag_out"

BANNER = ("D-020 FORMAL ablation — SafeSignal+Alsaify 둘 다 동일 정규화, "
          "A=both global / B=both per-lag")

SEEDS = [42, 43, 44]
ARMS = {
    "A_both_global": "checkpoints_track1_formal_global_s{s}",
    "B_both_perlag": "checkpoints_track1_formal_perlag_s{s}",
}
CLASSES = ["fall", "walking", "sit_stand", "lying", "standing", "picking"]
FALL = 0
BASELINE = {"fall_recall": 0.781, "far": 0.134}  # reference-only


def load_report(arm, seed):
    p = FT / ARMS[arm].format(s=seed) / "within_subject_test_report.json"
    if not p.exists():
        raise SystemExit(f"FAIL: report 없음: {p}")
    return json.load(open(p, encoding="utf-8"))


def tofall_rates(conf, counts):
    C = np.asarray(conf)
    out = {}
    for i, name in enumerate(CLASSES):
        if name == "fall":
            continue
        denom = counts.get(name, int(C[i].sum()))
        out[name] = float(C[i, FALL]) / denom if denom else 0.0
    return out


def msd(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def main():
    print("=" * 110)
    print(f"[ANALYZE] {BANNER}")
    print("=" * 110)

    data = {}
    for arm in ARMS:
        for s in SEEDS:
            r = load_report(arm, s); m = r["metrics"]
            data[(arm, s)] = {"threshold": r["threshold"], "fall_recall": m["fall_recall"],
                              "far": m["far"], "fall_f1": m["fall_f1"],
                              "confusion": m["confusion"], "counts": m["counts"],
                              "tofall": tofall_rates(m["confusion"], m["counts"])}

    # (a) 검증은 make_caches 에서 끝 — 여기선 A control reference 만
    a42 = data[("A_both_global", 42)]
    d_rec = a42["fall_recall"] - BASELINE["fall_recall"]
    d_far = a42["far"] - BASELINE["far"]
    print("\n## (a) A안(both global) seed42 — reference-only (gate-stop 아님)")
    print(f"  A_s42: fall_recall={a42['fall_recall']:.3f} (Δvs baseline={d_rec:+.3f})  "
          f"FAR={a42['far']:.3f} (Δ={d_far:+.3f})  worst|Δ|={max(abs(d_rec),abs(d_far)):.3f}")
    print("  (Alsaify 행순서가 baseline 과 달라 0.781 재현 더 어긋날 수 있음 → 참고만. 주 기준은 paired A-vs-B)")

    # (b) 3-seed mean±std
    print("\n## (b) arm별 3-seed best_operating (sealed test, mean±std)")
    agg = {}
    for arm in ARMS:
        rec = [data[(arm, s)]["fall_recall"] for s in SEEDS]
        far = [data[(arm, s)]["far"] for s in SEEDS]
        f1 = [data[(arm, s)]["fall_f1"] for s in SEEDS]
        agg[arm] = {"recall": msd(rec), "far": msd(far), "f1": msd(f1),
                    "recall_raw": rec, "far_raw": far, "f1_raw": f1}
        print(f"  [{arm}] fall_recall={agg[arm]['recall'][0]:.3f}±{agg[arm]['recall'][1]:.3f}  "
              f"FAR={agg[arm]['far'][0]:.3f}±{agg[arm]['far'][1]:.3f}  "
              f"fall_f1={agg[arm]['f1'][0]:.3f}±{agg[arm]['f1'][1]:.3f}")
        print(f"        per-seed recall={['%.3f'%v for v in rec]} far={['%.3f'%v for v in far]} "
              f"f1={['%.3f'%v for v in f1]} thr={[data[(arm,s)]['threshold'] for s in SEEDS]}")

    # paired recall/far/f1 delta (B-A)
    print("\n## paired delta (B-A, 동일 seed) — recall/FAR/F1")
    for key in ["fall_recall", "far", "fall_f1"]:
        d = [data[("B_both_perlag", s)][key] - data[("A_both_global", s)][key] for s in SEEDS]
        signs = {np.sign(round(x, 6)) for x in d}
        cons = len(signs) == 1 and 0 not in signs
        print(f"  {key:11s} ΔB-A 평균={np.mean(d):+.3f} per-seed={['%+.3f'%x for x in d]} 부호일관={'예' if cons else '아니오'}")

    # seed-합산 confusion
    print("\n## seed-합산(3 seeds) test confusion (행=true, 열=pred; fall,walk,sit,lie,stand,pick)")
    for arm in ARMS:
        Csum = np.zeros((6, 6), dtype=int)
        for s in SEEDS:
            Csum += np.asarray(data[(arm, s)]["confusion"], dtype=int)
        print(f"  [{arm}]")
        for i, name in enumerate(CLASSES):
            print(f"    {name:9s} {Csum[i].tolist()}")

    # (c) →fall + paired delta
    print("\n## (c) →fall 오분류율 + paired delta(B-A, 동일 seed)")
    tofall_summary = {}
    for cls in ["standing", "walking", "sit_stand", "picking"]:
        a_vals = [data[("A_both_global", s)]["tofall"][cls] for s in SEEDS]
        b_vals = [data[("B_both_perlag", s)]["tofall"][cls] for s in SEEDS]
        deltas = [b - a for a, b in zip(a_vals, b_vals)]
        signs = {np.sign(round(d, 6)) for d in deltas}
        cons = len(signs) == 1 and 0 not in signs
        am, asd = msd(a_vals); bm, bsd = msd(b_vals); dm, _ = msd(deltas)
        tofall_summary[cls] = {"A": a_vals, "B": b_vals, "delta": deltas,
                               "delta_mean": dm, "sign_consistent": cons}
        tag = " ← 핵심" if cls == "standing" else (" ← trade 감시" if cls == "walking" else "")
        print(f"  {cls:9s}→fall: A={am:.3f}±{asd:.3f}  B={bm:.3f}±{bsd:.3f}  "
              f"ΔB-A 평균={dm:+.3f} per-seed={['%+.3f'%d for d in deltas]} "
              f"부호일관={'예' if cons else '아니오'}{tag}")

    # (d) 판정
    st = tofall_summary["standing"]; wk = tofall_summary["walking"]
    st_down = all(d < 0 for d in st["delta"])        # standing 3-seed 일관 감소
    wk_up = all(d > 0 for d in wk["delta"])           # walking 3-seed 일관 악화
    other_worse = any(tofall_summary[c]["sign_consistent"] and tofall_summary[c]["delta_mean"] > 0
                      for c in ["walking", "sit_stand", "picking"])
    b_far = agg["B_both_perlag"]["far"][0]
    far_ok = b_far <= 0.15
    print("\n## (d) per-lag 채택 판정 (트랙1 = 정식 결론 채택 가능)")
    print(f"  standing→fall 일관감소={st_down}(Δ={['%+.3f'%d for d in st['delta']]})  "
          f"walking→fall 일관악화={wk_up}(Δ={['%+.3f'%d for d in wk['delta']]})  "
          f"B FAR={b_far:.3f}(≤0.15:{far_ok})")
    if st_down and not other_worse and far_ok:
        verdict = "채택(ADOPT) — per-lag 가 standing→fall 일관 감소 + 타 오탐 비악화 + FAR≤0.15."
    elif st_down and other_worse:
        verdict = ("보류(HOLD/trade-off) — standing→fall 는 일관 개선되나 walking 등 다른 →fall 오탐이 "
                   "일관 악화. trade-off → 추가 튜닝 또는 기각 판단 필요.")
    else:
        verdict = ("기각(REJECT) — standing→fall 개선이 일관적이지 않거나 무변동/악화. "
                   "global 유지가 정식 결론.")
    print(f"  => {verdict}")
    print("  (트랙1은 mixed-norm 핸디캡 없음 → positive/negative 모두 정식 결론으로 채택 가능. "
          "warm-start mismatch 잔여 비대칭은 트랙2보다 훨씬 작음.)")

    out = {"banner": BANNER, "baseline_reference_only": BASELINE,
           "A_s42_reference": {"recall": a42["fall_recall"], "far": a42["far"]},
           "agg": {a: {"recall": agg[a]["recall"], "far": agg[a]["far"], "f1": agg[a]["f1"],
                       "recall_raw": agg[a]["recall_raw"], "far_raw": agg[a]["far_raw"],
                       "f1_raw": agg[a]["f1_raw"]} for a in ARMS},
           "tofall": tofall_summary, "verdict": verdict,
           "per_run": {f"{a}_s{s}": {k: data[(a, s)][k] for k in
                       ("threshold", "fall_recall", "far", "fall_f1", "tofall")}
                       for a in ARMS for s in SEEDS}}
    p = DIAG / "track1_formal_comparison.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[summary] {p}")
    print("=" * 110)
    print(f"[ANALYZE DONE] {BANNER}")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
