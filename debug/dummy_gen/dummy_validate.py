"""더미 검증 리포트 (read-only). 8.py 검증 재사용 + onset/splice 검증 추가.

- 통계: dummy.mean / origin.mean ratio ∈ [0.5,2.0]
- PCA coverage: origin이 dummy 분포(5~95%) 내 포함률 (>=0.80)
- subtype/profile/status 분포, onset_delta, baseline_noise, crop_oob
- splice 경계: warp 후 이동한 경계(100,300→f스케일)에서 새 불연속(artifact) 생겼는지
  (dummy 경계 jump vs origin 경계 jump 비)
- 샘플 시각화
입력: out/dummies_clean400.npz + out/lineage.csv. 산출: out/validation/.
"""
from __future__ import annotations
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "debug/dummy_gen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import dummy_clean400_lib as L  # noqa: E402

CLEANED = ROOT / "data/cleaned"
SPLICE = (100, 300)
CLEAN_LEN = 400


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="out")
    args = ap.parse_args()
    OUT = ROOT / "debug/dummy_gen" / args.subdir
    VDIR = OUT / "validation"
    VDIR.mkdir(parents=True, exist_ok=True)
    d = np.load(OUT / "dummies_clean400.npz", allow_pickle=True)
    X = d["X"]; names = [str(x) for x in d["aug_filename"]]
    status = {n: str(s) for n, s in zip(names, d["aug_status"])}
    lin = {r["aug_filename"]: r for r in csv.DictReader(open(OUT / "lineage.csv", encoding="utf-8-sig"))}
    print(f"[입력] dummies {len(names)} | use {sum(1 for s in status.values() if s=='use')}")

    # origin clean400 mean/feature 캐시 (PCA·stats용)
    origins = sorted({lin[n]["origin_filename"] for n in names})
    print(f"[origin clean400 캐싱] {len(origins)} (resample)")
    o_clean, o_mean, o_feat = {}, {}, {}
    for fn in origins:
        p = next(CLEANED.rglob(fn), None)
        if p is None:
            continue
        c4 = L.build_clean400(p)
        if c4 is None:
            continue
        o_clean[fn] = c4
        o_mean[fn] = float(np.mean(c4))
        o_feat[fn] = np.concatenate([c4.mean(axis=0), c4.std(axis=0)])

    # ── ① 통계 범위 ───────────────────────────────────────────────────────────
    ratios = []
    for i, n in enumerate(names):
        ofn = lin[n]["origin_filename"]
        if ofn not in o_mean or o_mean[ofn] == 0:
            continue
        ratios.append(float(np.mean(X[i]) / o_mean[ofn]))
    ratios = np.array(ratios)
    in_range = np.mean((ratios >= 0.5) & (ratios <= 2.0)) * 100
    print(f"\n① 통계 ratio(dummy/origin): median={np.median(ratios):.3f} "
          f"[{np.percentile(ratios,5):.2f},{np.percentile(ratios,95):.2f}] | [0.5,2.0] 내 {in_range:.1f}%")

    # ── ② PCA coverage ────────────────────────────────────────────────────────
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(42)
    of = np.array([o_feat[f] for f in origins if f in o_feat])
    didx = rng.choice(len(names), min(150, len(names)), replace=False)
    df = np.array([np.concatenate([X[i].mean(axis=0), X[i].std(axis=0)]) for i in didx])
    proj = PCA(n_components=2).fit_transform(np.vstack([of, df]))
    op, dp = proj[:len(of)], proj[len(of):]
    lo, hi = np.percentile(dp[:, 0], [5, 95])
    cov = np.mean((op[:, 0] >= lo) & (op[:, 0] <= hi)) * 100
    print(f"② PCA coverage(원본이 더미 5~95% 내): {cov:.1f}% (>=80 양호)")
    _, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(op[:, 0], op[:, 1], c="steelblue", s=25, alpha=0.6, label=f"origin({len(of)})")
    ax.scatter(dp[:, 0], dp[:, 1], c="tomato", s=15, alpha=0.4, label=f"dummy({len(dp)})")
    ax.legend(); ax.set_title("PCA origin vs dummy (clean400)"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(VDIR / "pca.png", dpi=130); plt.close()

    # ── ③ splice 경계 불연속 (warp 후 새 artifact?) ──────────────────────────
    def bnd_jump(arr, b):
        b = int(round(b))
        if b < 2 or b >= len(arr) - 2:
            return None
        diff = np.abs(np.diff(arr, axis=0)).mean(axis=1)  # frame간 변화
        return float(diff[b - 1:b + 1].max()), float(np.percentile(diff, 95))
    ratios_b = []
    for i, n in enumerate(names):
        if status[n] != "use":
            continue
        ofn = lin[n]["origin_filename"]
        if ofn not in o_clean:
            continue
        f = float(lin[n]["warp_factor"]); off = int(lin[n]["crop_offset"])
        new_len = max(2, int(CLEAN_LEN * f))
        for b in SPLICE:
            nb = round(b * (new_len - 1) / (CLEAN_LEN - 1)) - off
            dj = bnd_jump(X[i], nb)
            oj = bnd_jump(o_clean[ofn], b)
            if dj and oj and oj[0] > 0:
                ratios_b.append(dj[0] / oj[0])  # dummy경계 jump / origin경계 jump
    ratios_b = np.array(ratios_b)
    if len(ratios_b):
        worse = np.mean(ratios_b > 2.0) * 100
        print(f"③ splice 경계 jump 비(dummy/origin): median={np.median(ratios_b):.2f} "
              f"p95={np.percentile(ratios_b,95):.2f} | >2배(새 artifact 의심) {worse:.1f}%")

    # ── ④ baseline_noise / onset_delta / crop_oob (lineage) ──────────────────
    bn = [float(r["baseline_noise_ratio"]) for r in lin.values() if r["baseline_noise_ratio"] != ""]
    o_bn = []
    for fn in origins:
        if fn in o_clean:
            o_bn.append(L.detect_onset_clean(o_clean[fn])["baseline_noise_ratio"])
    print(f"④ baseline_noise_ratio: dummy median={np.median(bn):.3f} p90={np.percentile(bn,90):.3f} | "
          f"origin median={np.median(o_bn):.3f} → 증가 {np.median(bn)-np.median(o_bn):+.3f}")
    print(f"   crop_out_of_bounds: {sum(1 for r in lin.values() if r['reject_or_pending_reason']=='crop_out_of_bounds')}")
    print(f"   baseline_noise_high pending: {sum(1 for r in lin.values() if r['reject_or_pending_reason']=='baseline_noise_high')}")
    ud = [abs(int(r["onset_delta"])) for r in lin.values() if r["aug_status"] == "use" and r["onset_delta"] != ""]
    print(f"   use onset_delta(|·|): median={np.median(ud):.1f} p90={np.percentile(ud,90):.1f} max={max(ud)}")

    # ── ⑤ 샘플 시각화 (origin vs use dummy 3) ────────────────────────────────
    sample_origin = next((f for f in origins if f in o_clean), None)
    if sample_origin:
        rel = [i for i, n in enumerate(names) if lin[n]["origin_filename"] == sample_origin and status[n] == "use"][:3]
        _, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(o_clean[sample_origin].mean(axis=1), "k", lw=1.2); axes[0].set_title(f"origin {sample_origin} (clean400 mean)")
        for i in rel:
            axes[1].plot(X[i].mean(axis=1), lw=0.9, alpha=0.7, label=lin[names[i]]["transform_profile"])
        for b in SPLICE:
            axes[0].axvline(b, color="orange", ls="--", lw=0.8); axes[1].axvline(b, color="orange", ls="--", lw=0.8)
        axes[1].legend(fontsize=7); axes[1].set_title("use dummies"); axes[1].set_xlabel("clean frame")
        plt.tight_layout(); plt.savefig(VDIR / "sample.png", dpi=130); plt.close()
        print(f"⑤ 시각화: {VDIR}/pca.png, sample.png")

    # 요약
    print("\n=== 검증 요약 ===")
    st = Counter(status.values())
    print(f"분류: use {st.get('use',0)} / pending {st.get('pending',0)} / exclude {st.get('exclude',0)}")
    ok = in_range >= 95 and cov >= 80
    print(f"통계≥95%({in_range:.1f}) & PCA≥80%({cov:.1f}) → {'OK' if ok else '재검토'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
