"""Event-centered windowing 사전 게이트 진단 (read-only; diag_out/ 산출만).

BANNER: EVENT-CENTERED pre-gate diagnostic — raw SDP energy vs z-scored attention,
        threshold 관찰만(확정 금지), Alsaify 참고용.

검증 질문:
  Gate A: fall TP 와 non-fall FP 가 peak/post-pre 변동으로 갈리는가?
  Gate B: fall 과 다른 패턴의 triggered non-fall(hard negative)을 충분히 만들 수 있나?
  Gate C: 기존 300-frame/28-step 안에 FP 를 가를 신호가 있나?

기준 분리:
  - energy/peak/post-pre/height = raw SDP(z-score 전), safesignal_*_rawsdp.npz.
  - attention = z-scored 입력(per-window global z-score) 기준, model(x, return_attention=True)→(N,28).
  - raw energy / z-scored input / attention / y_true / y_pred 는 동일 derived row order.
소스/캐시/체크포인트/STATE 수정 없음. threshold 확정 없음(후보 분위수만).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.finetune.train import (
    SafeSignalDataset, split_safesignal_within_subject,
    predict_with_fall_threshold, SIDE_FALL_MARKERS, SOURCE_SAFESIGNAL,
)
from model.pretrained.model import CNNGRUAttention

BANNER = ("EVENT-CENTERED pre-gate diagnostic — raw SDP energy vs z-scored attention, "
          "threshold 관찰만(확정 금지), Alsaify 참고용")

CACHE = PROJECT_ROOT / "model" / "finetune" / "cache"
SS_RAW = CACHE / "safesignal_e1234_finetune7_rawsdp.npz"
PRET6 = CACHE / "safesignal_e1234_pretrained6.npz"      # allclose sanity 기준
GLOBAL6 = CACHE / "track1_ss_global6.npz"               # from_npz 권위 dataset (z-scored)
CKPT_DIR = PROJECT_ROOT / "model" / "finetune" / "checkpoints_compare6_cpu"
BEST = CKPT_DIR / "best_operating.pt"
REPORT = CKPT_DIR / "within_subject_test_report.json"
DIAG = PROJECT_ROOT / "debug" / "modeling" / "diag_out"
OUTDIR = DIAG / "event_gate"

CLASSES6 = ["fall", "walking", "sit_stand", "lying", "standing", "picking"]
RUNNING_IDX, PICKING_IDX = 5, 6
EPS = 1e-6


def banner(tag):
    print("=" * 104); print(f"[{tag}] {BANNER}"); print("=" * 104)


def gzscore(X):
    """per-window global z-score (모델 입력)."""
    m = X.mean(axis=(2, 3), keepdims=True); s = X.std(axis=(2, 3), keepdims=True)
    return ((X - m) / (s + EPS)).astype(np.float32)


def step_energy(sdp):  # sdp (28,20) raw -> (28,) per-step mean|.|
    return np.abs(sdp).mean(axis=1)


def post_pre_ratio(curve):
    """peak step 기준 pre/post 평균 energy ratio = post/pre. peak 포함 안 함."""
    pk = int(np.argmax(curve))
    pre = curve[:pk]; post = curve[pk + 1:]
    pm = float(pre.mean()) if pre.size else np.nan
    qm = float(post.mean()) if post.size else np.nan
    ratio = (qm / pm) if (pre.size and post.size and pm > 0) else np.nan
    return pk, pm, qm, ratio


def main():
    banner("EVENT-GATE START")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"banner": BANNER}

    # ── 0) threshold assert + 실제 성능 기록 ─────────────────────────────────
    ck = torch.load(BEST, map_location="cpu", weights_only=False)
    rep = json.load(open(REPORT, encoding="utf-8"))
    thr_ck = float(ck["threshold"]); thr_rep = float(rep["threshold"])
    print(f"[threshold] best_operating.pt={thr_ck}  report={thr_rep}")
    if abs(thr_ck - thr_rep) > 1e-12:
        raise SystemExit(f"FAIL: threshold mismatch ckpt={thr_ck} report={thr_rep} — 진단 중단")
    THR = thr_ck
    rm = rep["metrics"]
    print(f"[perf/report] threshold={THR} R={rm['fall_recall']:.3f} F1={rm['fall_f1']:.3f} "
          f"FAR={rm['far']:.3f}  (0.781 가정 금지)")
    out["threshold"] = THR
    out["report_perf"] = {"fall_recall": rm["fall_recall"], "fall_f1": rm["fall_f1"], "far": rm["far"]}

    # ── 1) raw 7-class → 6-class derive (row order/provenance 보존) ───────────
    raw = np.load(SS_RAW, allow_pickle=True)
    X_raw7 = raw["X"].astype(np.float32)
    y7 = raw["y"].astype(np.int64)
    keep7 = y7 != RUNNING_IDX
    orig_idx = np.arange(len(y7))[keep7]
    y6 = y7[keep7].copy(); y6[y6 == PICKING_IDX] = 5
    X_raw6 = X_raw7[keep7]
    fn6 = raw["filename"].astype(str)[keep7]
    act6 = raw["activity"].astype(str)[keep7]
    trial6 = raw["trial"].astype(np.int64)[keep7]
    subj6 = raw["subject"].astype(np.int64)[keep7]
    env6 = raw["env"].astype(np.int64)[keep7]
    wfi6 = raw["within_file_index"].astype(np.int64)[keep7]
    n6 = len(y6)
    print(f"[derive6] raw7={len(y7)} → raw6={n6} (running 제거, picking→5)  "
          f"label_range=[{y6.min()},{y6.max()}]")
    assert n6 == 3041, f"derived 6-class row count {n6} != 3041"

    # ── 2) z-scored X + allclose sanity vs 기존 pretrained6 ──────────────────
    X_z6 = gzscore(X_raw6)
    pre = np.load(PRET6, allow_pickle=True)
    pre_fn = pre["filename"].astype(str); pre_X = pre["X"].astype(np.float32)
    meta_ok = (len(pre_fn) == n6 and np.array_equal(pre_fn, fn6)
               and np.array_equal(pre["activity"].astype(str), act6)
               and np.array_equal(pre["trial"].astype(np.int64), trial6)
               and np.array_equal(pre["subject"].astype(np.int64), subj6)
               and np.array_equal(pre["env"].astype(np.int64), env6))
    allclose_x = bool(np.allclose(X_z6, pre_X, atol=1e-4, rtol=1e-4))
    amax = float(np.max(np.abs(X_z6 - pre_X)))
    print(f"[sanity-allclose] meta(filename/activity/trial/subject/env row-order)={meta_ok}  "
          f"X_z6 vs pretrained6 allclose(1e-4)={allclose_x} max_abs={amax:.3e}")
    if not (meta_ok and allclose_x):
        raise SystemExit("FAIL: derived raw→global z-score 가 기존 pretrained6 와 불일치 — 중단")
    out["sanity_allclose"] = {"meta_row_order": meta_ok, "X_allclose": allclose_x, "max_abs": amax}

    # ── 3) side-fall 필터 + 권위 dataset(from_npz) 정합 + split 재현 ──────────
    keep_sf = np.array([not any(mk in Path(n).stem.upper() for mk in SIDE_FALL_MARKERS)
                        for n in fn6], dtype=bool)
    n_excl = int((~keep_sf).sum())
    print(f"[side-fall] excluded={n_excl} (markers={SIDE_FALL_MARKERS})  kept={int(keep_sf.sum())}")
    # 권위 dataset (train.py from_npz: 동일 side-fall 필터 적용)
    ds = SafeSignalDataset.from_npz(GLOBAL6)
    # 내 병렬 raw 배열도 동일 필터 후 정렬 일치 검증
    k = keep_sf
    fn_k = fn6[k]; y_k = y6[k]
    assert list(map(str, ds.filenames)) == list(fn_k), "from_npz filenames != my kept order"
    assert np.array_equal(np.asarray(ds.y, dtype=np.int64), y_k), "from_npz y != my kept y"
    # raw/z/meta 정렬본 (ds row j ↔ 아래 *_k[j])
    Xraw_k = X_raw6[k]; Xz_k = X_z6[k]; act_k = act6[k]; subj_k = subj6[k]; wfi_k = wfi6[k]
    tr_k, val_k, te_k = split_safesignal_within_subject(ds, val_ratio=0.2, test_ratio=0.2, seed=42)
    test_idx = np.asarray(te_k.indices, dtype=int)
    print(f"[split] seed42 val0.2 test0.2 → train={len(tr_k.indices)} val={len(val_k.indices)} test={len(test_idx)}")
    # test counts sanity vs report
    te_counts = {CLASSES6[i]: int((y_k[test_idx] == i).sum()) for i in range(6)}
    rep_counts = rm["counts"]
    counts_ok = all(te_counts[c] == rep_counts.get(c, -1) for c in CLASSES6)
    print(f"[sanity-counts] test counts={te_counts}  report={rep_counts}  일치={counts_ok}")
    out["split"] = {"train": len(tr_k.indices), "val": len(val_k.indices), "test": int(len(test_idx)),
                    "test_counts": te_counts, "report_counts": rep_counts, "counts_match": counts_ok}

    # ── 4) best_operating.pt 재추론 (전체 row, attention 포함) ───────────────
    model = CNNGRUAttention(n_classes=6).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    Xz_all = np.concatenate([Xz_k], axis=0)  # (n_kept,1,28,20)
    probs = np.empty((len(Xz_k), 6), dtype=np.float32)
    attn = np.empty((len(Xz_k), 28), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(Xz_k), 256):
            xb = torch.from_numpy(Xz_k[s:s + 256]).to(device)
            lg, w = model(xb, return_attention=True)
            probs[s:s + 256] = torch.softmax(lg, dim=1).cpu().numpy()
            attn[s:s + 256] = w.cpu().numpy()
    y_pred_all = predict_with_fall_threshold(probs, THR)
    # test confusion 재현 vs report
    yt = y_k[test_idx]; yp = y_pred_all[test_idx]
    conf = np.zeros((6, 6), dtype=int)
    for a, b in zip(yt, yp):
        conf[a, b] += 1
    rep_conf = np.asarray(rm["confusion"], dtype=int)
    conf_ok = np.array_equal(conf, rep_conf)
    tp = int(conf[0, 0]); fn = int(conf[0, 1:].sum()); fp = int(conf[1:, 0].sum())
    R = tp / max(tp + fn, 1); FAR = fp / max(int((yt != 0).sum()), 1)
    print(f"[reinfer] test confusion==report: {conf_ok}  (TP={tp} FN={fn} FP={fp} R={R:.3f} FAR={FAR:.3f})")
    if not conf_ok:
        print("  [recomputed]\n" + "\n".join("   " + str(r) for r in conf.tolist()))
        print("  [report]\n" + "\n".join("   " + str(r) for r in rep_conf.tolist()))
        raise SystemExit("FAIL: split 재현 confusion != report — 중단")
    out["reinfer"] = {"confusion_match": conf_ok, "TP": tp, "FN": fn, "FP": fp,
                      "recall": R, "far": FAR}

    # ── 에너지 지표 (전체 kept row, raw 기준) ────────────────────────────────
    nK = len(y_k)
    peak_step = np.empty(nK, dtype=int)
    pre_e = np.empty(nK); post_e = np.empty(nK); ratio = np.empty(nK)
    peak_h = np.empty(nK)  # sdp_max_abs
    attn_peak = attn.argmax(axis=1)
    for i in range(nK):
        curve = step_energy(Xraw_k[i, 0])
        pk, pm, qm, rt = post_pre_ratio(curve)
        peak_step[i] = pk; pre_e[i] = pm; post_e[i] = qm; ratio[i] = rt
        peak_h[i] = float(np.abs(Xraw_k[i, 0]).max())

    def by_class(arr, mask=None):
        d = {}
        sel = np.ones(nK, bool) if mask is None else mask
        for ci, cn in enumerate(CLASSES6):
            v = arr[(y_k == ci) & sel]
            v = v[~np.isnan(v)] if v.dtype.kind == "f" else v
            if v.size:
                d[cn] = {"n": int(v.size), "mean": float(np.mean(v)), "median": float(np.median(v)),
                         "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
                         "p90": float(np.percentile(v, 90))}
        return d

    # ── 항목1: peak 위치 분포 ────────────────────────────────────────────────
    print("\n## 항목1 peak step 분포(0..27) by class [raw sdp_mean_abs]")
    peak_dist = {}
    for ci, cn in enumerate(CLASSES6):
        ps = peak_step[y_k == ci]
        hist, _ = np.histogram(ps, bins=np.arange(0, 29))
        peak_dist[cn] = {"n": int(ps.size), "median_step": float(np.median(ps)),
                         "mean_step": float(ps.mean()), "hist28": hist.tolist()}
        print(f"  {cn:9s} n={ps.size:4d} median_peak={np.median(ps):4.1f} mean={ps.mean():4.1f}")

    # ── 항목2: post/pre ratio (Gate A) ───────────────────────────────────────
    print("\n## 항목2 post/pre energy ratio by class [가설 fall<1, walking/picking≈1] (Gate A)")
    ratio_stats = by_class(ratio)
    for cn in CLASSES6:
        s = ratio_stats.get(cn)
        if s:
            print(f"  {cn:9s} n={s['n']:4d} mean={s['mean']:.3f} median={s['median']:.3f} "
                  f"p25={s['p25']:.3f} p75={s['p75']:.3f}")

    # ── 항목5: peak height 분리 ──────────────────────────────────────────────
    print("\n## 항목5 peak height(raw sdp_max_abs) by class")
    height_stats = by_class(peak_h)
    for cn in CLASSES6:
        s = height_stats.get(cn)
        if s:
            print(f"  {cn:9s} n={s['n']:4d} median={s['median']:.4f} p75={s['p75']:.4f} p90={s['p90']:.4f}")
    fall_h = peak_h[y_k == 0]
    fall_q = {q: float(np.percentile(fall_h, q)) for q in (50, 75, 90)}
    print(f"  fall peak_height 분위수 p50/p75/p90 = {fall_q[50]:.4f}/{fall_q[75]:.4f}/{fall_q[90]:.4f} "
          "(trigger 후보 범위 — 확정 아님)")

    # ── 항목4: hard negative 풀 (Gate B) ─────────────────────────────────────
    print("\n## 항목4 hard negative 풀: non-fall 중 fall-like peak height 윈도우 수 (activity별) (Gate B)")
    nonfall_mask = y_k != 0
    hard = {}
    for q in (50, 75, 90):
        thr_q = fall_q[q]
        sel = nonfall_mask & (peak_h >= thr_q)
        per_act = {}
        for a in sorted(set(act_k[sel].tolist())):
            cnt = int(((act_k == a) & sel).sum())
            # 이 hard negative 들이 fall 과 패턴이 다른지: post/pre ratio median
            rr = ratio[(act_k == a) & sel]; rr = rr[~np.isnan(rr)]
            per_act[a] = {"n": cnt, "ratio_median": float(np.median(rr)) if rr.size else None}
        hard[f"p{q}"] = {"thresh": thr_q, "total": int(sel.sum()), "by_activity": per_act}
        print(f"  fall p{q}({thr_q:.4f}) 초과 non-fall = {int(sel.sum())}개  "
              f"by_activity={ {a: per_act[a]['n'] for a in per_act} }")
    # fall 자신의 post/pre median (대조)
    fr = ratio[y_k == 0]; fr = fr[~np.isnan(fr)]
    print(f"  [대조] fall post/pre ratio median={np.median(fr):.3f}  "
          "→ hard negative ratio가 fall보다 크면(≈1) '패턴 다른 triggered non-fall' 충분")

    # ── 항목3: attention↔peak 정렬 (TP/FP/FN) ────────────────────────────────
    print("\n## 항목3 attention↔raw peak 정렬 (test set, 예측결과별)")
    groups = {
        "TP_fall": (test_idx[(yt == 0) & (yp == 0)]),
        "FN_fall": (test_idx[(yt == 0) & (yp != 0)]),
        "FP_walking": (test_idx[(y_k[test_idx] == 1) & (yp == 0)]),
        "FP_standing": (test_idx[(y_k[test_idx] == 4) & (yp == 0)]),
        "FP_picking": (test_idx[(y_k[test_idx] == 5) & (yp == 0)]),
        "FP_any_nonfall": (test_idx[(y_k[test_idx] != 0) & (yp == 0)]),
    }
    attn_align = {}
    for g, idx in groups.items():
        if len(idx) == 0:
            attn_align[g] = {"n": 0}; print(f"  {g:16s} n=0"); continue
        d_align = np.abs(attn_peak[idx] - peak_step[idx])
        a_pk = attn_peak[idx]
        attn_align[g] = {"n": int(len(idx)),
                         "attn_peak_median": float(np.median(a_pk)),
                         "energy_peak_median": float(np.median(peak_step[idx])),
                         "align_dist_median": float(np.median(d_align)),
                         "frac_exact_align": float((d_align == 0).mean()),
                         "frac_align_le2": float((d_align <= 2).mean())}
        a = attn_align[g]
        print(f"  {g:16s} n={a['n']:3d} attn_peak~{a['attn_peak_median']:.1f} "
              f"energy_peak~{a['energy_peak_median']:.1f} |Δ|~{a['align_dist_median']:.1f} "
              f"exact={a['frac_exact_align']:.2f} ≤2={a['frac_align_le2']:.2f}")

    # ── 항목6: fall subtype ──────────────────────────────────────────────────
    print("\n## 항목6 fall subtype별 (activity) peak/post-drop")
    subtype = {}
    fall_acts = sorted(set(act_k[y_k == 0].tolist()))
    if fall_acts:
        for a in fall_acts:
            m = (y_k == 0) & (act_k == a)
            rr = ratio[m]; rr = rr[~np.isnan(rr)]
            subtype[a] = {"n": int(m.sum()), "peak_step_median": float(np.median(peak_step[m])),
                          "postpre_ratio_median": float(np.median(rr)) if rr.size else None,
                          "peak_height_median": float(np.median(peak_h[m]))}
            print(f"  {a:14s} n={int(m.sum()):4d} peak~{np.median(peak_step[m]):.1f} "
                  f"post/pre~{(np.median(rr) if rr.size else float('nan')):.3f} "
                  f"height~{np.median(peak_h[m]):.4f}")
    else:
        subtype = "unavailable"; print("  subtype unavailable")

    # ── 항목7: peak-centered 300-frame sanity (1차 근사) ─────────────────────
    edge = np.minimum(peak_step, 27 - peak_step)  # 가장자리까지 거리
    fall_edge = edge[y_k == 0]
    print("\n## 항목7 peak-centered sanity (28-step 인덱스 근사)")
    print(f"  fall peak 중앙편향: edge_dist median={np.median(fall_edge):.1f} "
          f"(0=가장자리,14=중앙)  peak가 윈도우 끝쪽이면 centered window 재절단 이득 큼")
    print("  [주의] 실제 centered window 는 원본 CSV 300-frame 재절단+RPCA→SDP 재계산 필요 → "
          "이 진단(28-step)으로 완전 판정 불가.")

    # ── 그림 저장 ────────────────────────────────────────────────────────────
    def save_hist(data_by_class, title, fname, clip=None):
        plt.figure(figsize=(7, 4))
        for cn in CLASSES6:
            v = data_by_class.get(cn)
            if v is None or len(v) == 0:
                continue
            vv = v[~np.isnan(v)] if v.dtype.kind == "f" else v
            if clip:
                vv = np.clip(vv, *clip)
            plt.hist(vv, bins=30, alpha=0.45, label=cn, density=True)
        plt.title(title[:90]); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(OUTDIR / fname, dpi=90); plt.close()

    save_hist({c: ratio[y_k == i] for i, c in enumerate(CLASSES6)},
              "post/pre energy ratio by class (raw)", "ratio_by_class.png", clip=(0, 3))
    save_hist({c: peak_h[y_k == i] for i, c in enumerate(CLASSES6)},
              "peak height (raw sdp_max_abs) by class", "peakheight_by_class.png")
    save_hist({c: peak_step[y_k == i].astype(float) for i, c in enumerate(CLASSES6)},
              "peak step (0..27) by class", "peakstep_by_class.png")
    # fall vs non-fall peak height overlay
    plt.figure(figsize=(7, 4))
    plt.hist(peak_h[y_k == 0], bins=40, alpha=0.5, density=True, label="fall")
    plt.hist(peak_h[y_k != 0], bins=40, alpha=0.5, density=True, label="non-fall")
    for q in (50, 75, 90):
        plt.axvline(fall_q[q], ls="--", lw=0.8, color="k")
    plt.title("fall vs non-fall peak height (raw) + fall p50/75/90"); plt.legend()
    plt.tight_layout(); plt.savefig(OUTDIR / "peakheight_fall_vs_nonfall.png", dpi=90); plt.close()

    # ── 게이트 판정 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("## 게이트 판정")
    # Gate A: fall TP vs non-fall FP post/pre 분리
    fall_ratio_med = float(np.nanmedian(ratio[y_k == 0]))
    nonfall_ratio_med = float(np.nanmedian(ratio[y_k != 0]))
    # FP 윈도우 ratio (test)
    fp_idx = test_idx[(y_k[test_idx] != 0) & (yp == 0)]
    tp_idx = test_idx[(yt == 0) & (yp == 0)]
    fp_ratio_med = float(np.nanmedian(ratio[fp_idx])) if len(fp_idx) else float("nan")
    tp_ratio_med = float(np.nanmedian(ratio[tp_idx])) if len(tp_idx) else float("nan")
    sep_ratio = nonfall_ratio_med - fall_ratio_med
    gate_a = "긍정" if sep_ratio > 0.15 else ("약함" if sep_ratio > 0.05 else "부정")
    print(f"  Gate A (peak/post-pre 분리): fall ratio med={fall_ratio_med:.3f} vs "
          f"non-fall={nonfall_ratio_med:.3f} (Δ={sep_ratio:+.3f}); "
          f"TP ratio={tp_ratio_med:.3f} vs FP ratio={fp_ratio_med:.3f} → {gate_a}")
    # Gate B: hard negative 충분성 (fall p75 초과 non-fall 수 + 패턴 차이)
    hn75 = hard["p75"]["total"]
    gate_b = "긍정" if hn75 >= 200 else ("약함" if hn75 >= 50 else "부정")
    print(f"  Gate B (triggered non-fall 충분성): fall p75 초과 non-fall={hn75}개 "
          f"(p50={hard['p50']['total']}, p90={hard['p90']['total']}); "
          f"이들 ratio가 fall보다 큰지로 '패턴 다름' 판단 → {gate_b}")
    # Gate C: 현재 표현에 FP 가를 신호 — height 또는 ratio 또는 attention 분리
    h_fall_p50 = fall_q[50]; h_nonfall_med = float(np.median(peak_h[y_k != 0]))
    height_sep = h_fall_p50 - h_nonfall_med
    gate_c = "긍정" if (sep_ratio > 0.1 or height_sep > 0.02) else ("약함" if (sep_ratio > 0.03 or height_sep > 0) else "부정")
    print(f"  Gate C (300-frame/28-step 내 분리 신호): height Δ(fall p50 - nonfall med)={height_sep:+.4f}, "
          f"ratio Δ={sep_ratio:+.3f} → {gate_c}")

    go = (gate_a == "긍정" and gate_b in ("긍정", "약함")) or (gate_a in ("긍정", "약함") and gate_c == "긍정")
    overall = "GO" if go else "보류"
    blocked = [g for g, v in [("A", gate_a), ("B", gate_b), ("C", gate_c)] if v == "부정"]
    print(f"\n  종합: event-centered 본 구현 → {overall}"
          + (f"  (막은 게이트: {blocked})" if blocked else ""))
    print("  주의: peak-centered 실제 효과는 원본 CSV 재절단+RPCA→SDP 재계산 필요(항목7) — 28-step 근사 한계.")

    out.update({
        "peak_step_dist": peak_dist, "postpre_ratio": ratio_stats, "peak_height": height_stats,
        "fall_height_quantiles": fall_q, "hard_negative_pool": hard,
        "attn_peak_align": attn_align, "fall_subtype": subtype,
        "gates": {"A": gate_a, "B": gate_b, "C": gate_c, "overall": overall, "blocked": blocked,
                  "fall_ratio_med": fall_ratio_med, "nonfall_ratio_med": nonfall_ratio_med,
                  "tp_ratio_med": tp_ratio_med, "fp_ratio_med": fp_ratio_med,
                  "hard_neg_p75": hn75, "height_sep": height_sep},
    })
    (OUTDIR / "event_gate_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    # per-window CSV (raw energy + pred + attention peak)
    import csv
    with open(OUTDIR / "event_gate_windows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "within_file_index", "activity", "subject", "y_true", "y_pred",
                    "in_test", "peak_step", "attn_peak", "pre_e", "post_e", "postpre_ratio", "peak_height"])
        test_set = set(test_idx.tolist())
        for i in range(nK):
            w.writerow([fn_k[i], int(wfi_k[i]), act_k[i], int(subj_k[i]), int(y_k[i]),
                        int(y_pred_all[i]), int(i in test_set), int(peak_step[i]), int(attn_peak[i]),
                        f"{pre_e[i]:.6f}", f"{post_e[i]:.6f}",
                        ("" if np.isnan(ratio[i]) else f"{ratio[i]:.6f}"), f"{peak_h[i]:.6f}"])
    print(f"\n[out] {OUTDIR}/  (event_gate_summary.json, event_gate_windows.csv, *.png)")
    banner("EVENT-GATE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
