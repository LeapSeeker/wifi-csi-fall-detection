"""트랙2 PROVISIONAL probe — A(global control) vs B(per-lag mixed) 비교 분석.

BANNER: PROVISIONAL mixed-normalization probe — SafeSignal per-lag vs global,
        Alsaify=global 고정, D-020 정식 ablation 아님.

within_subject_test_report.json (sealed test, operating threshold) 만 사용한다.
history.json primary_val 은 7-class helper 라벨 혼동 위험으로 쓰지 않는다.
standing→fall 등 →fall 오분류율 = test confusion[true_row][fall_col=0] / counts[true].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FT = PROJECT_ROOT / "model" / "finetune"
DIAG = PROJECT_ROOT / "debug" / "modeling" / "diag_out"

BANNER = ("PROVISIONAL mixed-normalization probe — SafeSignal per-lag vs global, "
          "Alsaify=global 고정, D-020 정식 ablation 아님")

SEEDS = [42, 43, 44]
ARMS = {
    "A_global_control": "checkpoints_track2_probe_global_control_s{s}",
    "B_perlag_mixed":   "checkpoints_track2_probe_perlag_mixed_s{s}",
}
CLASSES = ["fall", "walking", "sit_stand", "lying", "standing", "picking"]
FALL = 0
BASELINE = {"fall_recall": 0.781, "far": 0.134}  # 기존 CPU baseline (seed42 단일 대응)


def load_report(arm: str, seed: int) -> dict:
    p = FT / ARMS[arm].format(s=seed) / "within_subject_test_report.json"
    if not p.exists():
        raise SystemExit(f"FAIL: report 없음: {p}")
    return json.load(open(p, encoding="utf-8"))


def tofall_rates(conf: list, counts: dict) -> dict:
    """각 non-fall true class → fall 오분류율 = conf[row][0]/counts[class]."""
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


def main() -> int:
    print("=" * 110)
    print(f"[ANALYZE] {BANNER}")
    print("=" * 110)

    data = {}  # (arm,seed) -> dict
    for arm in ARMS:
        for s in SEEDS:
            r = load_report(arm, s)
            m = r["metrics"]
            data[(arm, s)] = {
                "threshold": r["threshold"],
                "fall_recall": m["fall_recall"], "far": m["far"], "fall_f1": m["fall_f1"],
                "confusion": m["confusion"], "counts": m["counts"],
                "tofall": tofall_rates(m["confusion"], m["counts"]),
            }

    # ── (a) A안 seed42 control gate ──────────────────────────────────────────
    a42 = data[("A_global_control", 42)]
    d_rec = a42["fall_recall"] - BASELINE["fall_recall"]
    d_far = a42["far"] - BASELINE["far"]
    worst = max(abs(d_rec), abs(d_far))
    if worst <= 0.05:
        verdict = "정상(±0.05 이내) — 플로우 정합"
    elif worst <= 0.10:
        verdict = "경고(±0.05~0.10) — 플로우 OK이나 B안은 참고용으로만 해석"
    else:
        verdict = "중단(±0.10 초과) — raw→정규화→derive→학습 플로우 결함부터 점검 필요"
    print("\n## (a) A안(global control) seed42 게이트 [기존 CPU baseline 0.781/0.134 대응]")
    print(f"  A_s42: fall_recall={a42['fall_recall']:.3f} (Δ={d_rec:+.3f})  "
          f"FAR={a42['far']:.3f} (Δ={d_far:+.3f})  worst|Δ|={worst:.3f}")
    print(f"  => 판정: {verdict}")

    # ── (b) A/B 3-seed 평균±표준편차 ─────────────────────────────────────────
    print("\n## (b) arm별 3-seed best_operating 핵심지표 (sealed test, mean±std)")
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
    print("  (A안 3-seed 평균은 pass/fail gate 아님 — paired delta 해석용 분포 sanity only)")

    # ── seed-합산 confusion A/B 나란히 ───────────────────────────────────────
    print("\n## seed-합산(3 seeds) test confusion  (행=true, 열=pred; 순서=fall,walk,sit,lie,stand,pick)")
    for arm in ARMS:
        Csum = np.zeros((6, 6), dtype=int)
        for s in SEEDS:
            Csum += np.asarray(data[(arm, s)]["confusion"], dtype=int)
        print(f"  [{arm}]")
        for i, name in enumerate(CLASSES):
            print(f"    {name:9s} {Csum[i].tolist()}")

    # ── (c) →fall 오분류율 + paired delta(B-A) ──────────────────────────────
    print("\n## (c) →fall 오분류율 (test confusion) 및 paired delta(B-A, 같은 seed)")
    nonfall = [c for c in CLASSES if c != "fall"]
    for cls in ["standing", "walking", "sit_stand", "picking"]:
        a_vals = [data[("A_global_control", s)]["tofall"][cls] for s in SEEDS]
        b_vals = [data[("B_perlag_mixed", s)]["tofall"][cls] for s in SEEDS]
        deltas = [b - a for a, b in zip(a_vals, b_vals)]
        signs = {np.sign(round(d, 6)) for d in deltas}
        consistent = len(signs) == 1 and 0 not in signs
        am, asd = msd(a_vals); bm, bsd = msd(b_vals); dm, dsd = msd(deltas)
        tag = " ← 핵심(standing→fall, 기준 ~0.34 대비)" if cls == "standing" else ""
        print(f"  {cls:9s}→fall: A={am:.3f}±{asd:.3f}  B={bm:.3f}±{bsd:.3f}  "
              f"ΔB-A 평균={dm:+.3f} per-seed={['%+.3f'%d for d in deltas]} "
              f"부호일관={'예' if consistent else '아니오'}{tag}")

    # ── (d) 해석 판정 ────────────────────────────────────────────────────────
    st_a = [data[("A_global_control", s)]["tofall"]["standing"] for s in SEEDS]
    st_b = [data[("B_perlag_mixed", s)]["tofall"]["standing"] for s in SEEDS]
    st_d = [b - a for a, b in zip(st_a, st_b)]
    st_consistent_down = all(d < 0 for d in st_d)
    b_far_mean = agg["B_perlag_mixed"]["far"][0]
    far_ok = b_far_mean <= 0.15
    print("\n## (d) 해석 판정")
    print(f"  standing→fall ΔB-A per-seed={['%+.3f'%d for d in st_d]} "
          f"(일관 감소={'예' if st_consistent_down else '아니오'}), B FAR 평균={b_far_mean:.3f}(≤0.15:{far_ok})")
    if worst > 0.10:
        print("  => 게이트 중단 조건 — 플로우 결함 점검 우선. B 해석 보류.")
    elif st_consistent_down and far_ok and (np.mean(st_d) <= -0.02):
        print("  => B안이 standing→fall 를 3-seed 일관 감소 + FAR≤0.15: 트랙1(Alsaify도 per-lag) "
              "정식 ablation 투자 가치 있음(positive — 강하게 해석 가능).")
    else:
        print("  => B안 무변동/약함/악화: 약한 증거. mixed-norm(SafeSignal per-lag 60% 과대표집)+ "
              "best.pt(global) warm-start mismatch 이중 핸디캡 → probe 단독으로 per-lag 기각 금지. "
              "정식 결론은 트랙1 결과로만. (negative 는 보수적으로 해석)")
    if 0.05 < worst <= 0.10:
        print("  주의: A control 이 경고밴드 → 위 B 해석은 참고용(reference-only).")

    # ── 산출물 저장 ──────────────────────────────────────────────────────────
    out = {
        "banner": BANNER,
        "baseline": BASELINE,
        "control_gate": {"A_s42_recall": a42["fall_recall"], "A_s42_far": a42["far"],
                         "d_recall": d_rec, "d_far": d_far, "worst_abs": worst, "verdict": verdict},
        "agg": {arm: {"recall_mean_std": agg[arm]["recall"], "far_mean_std": agg[arm]["far"],
                      "f1_mean_std": agg[arm]["f1"], "recall_raw": agg[arm]["recall_raw"],
                      "far_raw": agg[arm]["far_raw"], "f1_raw": agg[arm]["f1_raw"]} for arm in ARMS},
        "tofall": {cls: {"A": [data[("A_global_control", s)]["tofall"][cls] for s in SEEDS],
                         "B": [data[("B_perlag_mixed", s)]["tofall"][cls] for s in SEEDS]}
                   for cls in nonfall},
        "per_run": {f"{arm}_s{s}": {k: data[(arm, s)][k] for k in
                    ("threshold", "fall_recall", "far", "fall_f1", "tofall")}
                    for arm in ARMS for s in SEEDS},
    }
    DIAG.mkdir(parents=True, exist_ok=True)
    p = DIAG / "track2_probe_comparison.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[summary] {p}")
    print("=" * 110)
    print(f"[ANALYZE DONE] {BANNER}")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
