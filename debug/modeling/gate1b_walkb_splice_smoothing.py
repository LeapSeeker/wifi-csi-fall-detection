"""게이트 1-b 후속: WALK_B splice-smoothing probe (read-only).

목적: concat_main 의 WALK_B fall 신호가 frame 50 근처(=onset=splice)에 의존하는 것이
  (A) beep 제거로 생긴 합성 sharp discontinuity(splice artifact) 때문인가
  (B) 실제 fall onset/motion transition 보존 때문인가 를 가른다.

핵심 원칙 (반드시 준수):
  smoothing 은 RAW AMPLITUDE crop 레벨에 적용하고, 그 후 정식 파이프라인을 다시 태운다.
    raw amplitude crop (300, n_sc)
    → [splice junction smoothing on raw amplitude]
    → rpca_sparse → stacked_doppler_profile → global z-score
    → window_to_model_input → (1,28,20) → (1,1,28,20) → finetuned_baseline6
  z-scored SDP row 를 직접 건드리면 row occlusion 과 동일해져 새 정보가 없다.

대상 3세션 (WALK_B, concat-only thr0.1 crossing):
  E1_S01_A_FALL_WALK_B_T001 (concat 0.195 / center 0.019)
  E2_S02_A_FALL_WALK_B_T001 (concat 0.924 / center 0.015)  ← 핵심 판정 케이스
  E3_S03_A_FALL_WALK_B_T001 (concat 0.295 / center 0.046)

좌표: concat_main = clean[50:350] = orig[100:150]+[200:400]+[450:500]
  junction(crop-local): 50 (=orig149→200, onset/splice), 250 (=orig399→450, offset/splice)

설계 (분리 실행):
  boundary_condition: boundary_50_only | boundary_250_only | both_boundaries
  width(강도):         3, 5, 10  frames
  method:              A_linear (선형 crossfade) | B_movavg (이동평균)  — 둘 다 raw amplitude
판정 결정타: boundary_50 smoothing 후 fall_prob 가 살아남느냐 붕괴하느냐.
  continuous_center 수렴은 보조(crop 내용이 onset 외에도 달라 완전 수렴 안 할 수 있음).

제약: 동결 파일(pipeline/rpca/acf/sdp/학습·추론) 수정 금지, 새 학습 금지, 데이터 read-only.
      산출은 debug/modeling/diag_out/beep_concat_artifact/gate1b_smoothing/ 하위.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 형제 모듈 import 용

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from model.preprocessing.loader import load_safesignal_csv          # noqa: E402
from model.preprocessing.resample import resample_to_100hz          # noqa: E402

# 검증된 헬퍼 재사용 (모델 로드/추론/전처리/좌표) — 게이트 1-b occlusion 모듈
from gate1b_walkb_occlusion import (                                # noqa: E402
    load_model, infer, preprocess_crop, build_crops, find_file, att_block,
    row_energy, BOUNDARY_50_ROWS, BOUNDARY_250_ROWS,
    CKPT, CKPT_ID, PRETRAINED6_CLASSES, FALL_IDX, TARGETS,
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

OUT = ROOT / "debug/modeling/diag_out/beep_concat_artifact/gate1b_smoothing"
THR = 0.1

BOUNDARY_CONDITIONS = {
    "boundary_50_only": [50],
    "boundary_250_only": [250],
    "both_boundaries": [50, 250],
}
WIDTHS = [3, 5, 10]
METHODS = ["A_linear", "B_movavg"]


# ─── raw amplitude smoothing ─────────────────────────────────────────────────
def smooth_linear(crop, b, w):
    """method A: 선형 crossfade. 윈도우 [b-w, b+w) 를 바깥 anchor crop[b-w-1], crop[b+w]
    사이 선형 보간으로 대체 → discontinuity(step) 제거. raw amplitude 레벨, 전 서브캐리어."""
    out = np.array(crop, dtype=np.float64, copy=True)
    N = len(crop)
    lo, hi = b - w, b + w
    la = max(0, lo - 1)
    ra = min(N - 1, hi)
    left = out[la]
    right = out[ra]
    span = ra - la
    if span <= 0:
        return out.astype(crop.dtype, copy=False)
    for i in range(max(0, lo), min(N, hi)):
        alpha = (i - la) / span
        out[i] = (1.0 - alpha) * left + alpha * right
    return out.astype(crop.dtype, copy=False)


def smooth_movavg(crop, b, w):
    """method B: 이동평균. 윈도우 [b-w, b+w) 의 각 프레임을 원본 crop 의 중심 이동평균
    (반경 w, 커널 2w+1)으로 대체 → step 을 ramp 로 완화. raw amplitude 레벨."""
    src = np.asarray(crop, dtype=np.float64)
    out = np.array(crop, dtype=np.float64, copy=True)
    N = len(crop)
    lo, hi = b - w, b + w
    for i in range(max(0, lo), min(N, hi)):
        a = max(0, i - w)
        bb = min(N, i + w + 1)
        out[i] = src[a:bb].mean(axis=0)
    return out.astype(crop.dtype, copy=False)


def apply_smoothing(crop, method, bs, w):
    fn = smooth_linear if method == "A_linear" else smooth_movavg
    out = crop
    for b in bs:
        out = fn(out, b, w)
    return out


def amp_jump(crop, b):
    """접합부 인접 프레임 평균 절대차 = mean(|crop[b]-crop[b-1]|)."""
    return float(np.abs(crop[b] - crop[b - 1]).mean())


# ─── crop → 모델 fall_prob + attention ───────────────────────────────────────
def crop_infer(crop, model, device):
    pp = preprocess_crop(crop)
    mi = pp["model_input"]                 # (1,28,20)
    p, attn = infer(model, mi, device, want_attention=True)
    re_z = row_energy(pp["sdp_z"])
    _, b50 = att_block(attn, BOUNDARY_50_ROWS)
    _, b250 = att_block(attn, BOUNDARY_250_ROWS)
    return {
        "fall_prob": float(p[FALL_IDX]),
        "pred_class": PRETRAINED6_CLASSES[int(p.argmax())],
        "attn_b50_ratio": b50, "attn_b250_ratio": b250,
        "attn_peak_row": int(np.argmax(attn)),
        "sdp_peak_row": int(re_z.argmax()),
    }


# ─── 세션 분석 ───────────────────────────────────────────────────────────────
def analyze(stem, model, device):
    path = find_file(stem)
    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    A550 = res.amplitude[:550]
    crops = build_crops(A550)
    concat = crops["concat_main"]
    center = crops["continuous_center"]

    base = crop_infer(concat, model, device)
    center_info = crop_infer(center, model, device)
    fall_orig = base["fall_prob"]
    center_fall = center_info["fall_prob"]
    jump_orig = {50: amp_jump(concat, 50), 250: amp_jump(concat, 250)}

    rows = []
    for method in METHODS:
        for cond, bs in BOUNDARY_CONDITIONS.items():
            for w in WIDTHS:
                sm = apply_smoothing(concat, method, bs, w)
                inf = crop_infer(sm, model, device)
                jump_after = {b: amp_jump(sm, b) for b in (50, 250)}
                rows.append({
                    "session": stem, "method": method, "boundary_condition": cond, "width": w,
                    "fall_prob_original_concat": fall_orig,
                    "fall_prob_smoothed": inf["fall_prob"],
                    "fall_prob_delta": fall_orig - inf["fall_prob"],
                    "pred_class": inf["pred_class"],
                    "crosses_thr_0p1": bool(inf["fall_prob"] >= THR),
                    "attention_boundary_50_ratio": inf["attn_b50_ratio"],
                    "attention_boundary_250_ratio": inf["attn_b250_ratio"],
                    "attention_peak_row": inf["attn_peak_row"],
                    "sdp_row_energy_peak_row": inf["sdp_peak_row"],
                    "continuous_center_fall_prob": center_fall,
                    "distance_to_continuous_center": inf["fall_prob"] - center_fall,
                    "amp_jump_b50_orig": jump_orig[50], "amp_jump_b250_orig": jump_orig[250],
                    "amp_jump_b50_smoothed": jump_after[50], "amp_jump_b250_smoothed": jump_after[250],
                })

    # ── 세션 판정 (boundary_50_only, 작은 width 3/5, 두 method 평균) ──────────
    b50_small = [r for r in rows if r["boundary_condition"] == "boundary_50_only" and r["width"] in (3, 5)]
    mean_small = float(np.mean([r["fall_prob_smoothed"] for r in b50_small])) if b50_small else None
    # method 일관성: A/B 둘 다 같은 방향(붕괴/유지)인가
    a_small = np.mean([r["fall_prob_smoothed"] for r in b50_small if r["method"] == "A_linear"])
    b_small = np.mean([r["fall_prob_smoothed"] for r in b50_small if r["method"] == "B_movavg"])
    methods_agree = bool((a_small < THR) == (b_small < THR))
    # 붕괴 판정: 작은 width 평균 fall 이 thr 아래로 떨어지고 원본 대비 큰 폭 하락
    drop_frac = (fall_orig - mean_small) / fall_orig if (mean_small is not None and fall_orig > 1e-6) else 0.0
    if fall_orig < THR:
        verdict = "na_original_below_thr"
    elif mean_small < THR and drop_frac >= 0.5:
        verdict = "A_splice_artifact_leaning"   # b50 smoothing 만으로 붕괴
    elif mean_small >= THR and drop_frac < 0.5:
        verdict = "B_onset_preserved_leaning"   # b50 smoothing 후에도 유지
    else:
        verdict = "mixed"

    summary = {
        "session": stem,
        "fall_prob_original_concat": fall_orig,
        "original_pred_class": base["pred_class"],
        "continuous_center_fall_prob": center_fall,
        "original_attn_b50_ratio": base["attn_b50_ratio"],
        "original_attn_peak_row": base["attn_peak_row"],
        "amp_jump_b50_orig": jump_orig[50], "amp_jump_b250_orig": jump_orig[250],
        "boundary_50_small_width_mean_fall": mean_small,
        "drop_frac_small_width": drop_frac,
        "methods_agree": methods_agree,
        "verdict": verdict,
    }
    return rows, summary


# ─── 그림 (fall_prob vs width) ───────────────────────────────────────────────
def plot_session(stem, rows, summ, plots_dir):
    if not HAVE_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    styles = {"boundary_50_only": "-", "boundary_250_only": "--", "both_boundaries": ":"}
    colors = {"A_linear": "C0", "B_movavg": "C1"}
    for method in METHODS:
        for cond in BOUNDARY_CONDITIONS:
            pts = sorted([r for r in rows if r["method"] == method and r["boundary_condition"] == cond],
                         key=lambda r: r["width"])
            xs = [r["width"] for r in pts]
            ys = [r["fall_prob_smoothed"] for r in pts]
            ax.plot(xs, ys, styles[cond], color=colors[method], marker="o", ms=4,
                    label=f"{method} {cond}")
    ax.axhline(summ["fall_prob_original_concat"], color="0.3", lw=1.4, label="original concat")
    ax.axhline(summ["continuous_center_fall_prob"], color="0.6", ls=":", lw=1.2, label="continuous_center")
    ax.axhline(THR, color="C3", ls="--", lw=1.0, label=f"thr {THR}")
    ax.set_xlabel("smoothing width (frames)"); ax.set_ylabel("fall_prob (smoothed)")
    ax.set_ylim(-0.02, 1.02); ax.set_xticks(WIDTHS)
    ax.set_title(f"{stem} | splice-smoothing → fall_prob  (verdict: {summ['verdict']})", fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(plots_dir / f"{stem}__smoothing_fallprob.png", dpi=120); plt.close(fig)


# ─── share 텍스트 ────────────────────────────────────────────────────────────
def build_share(all_rows, summaries):
    L = ["# 게이트 1-b WALK_B splice-smoothing probe\n",
         f"checkpoint: {CKPT_ID}  |  smoothing: RAW amplitude → 정식 RPCA→SDP→모델 재실행",
         f"boundary_condition: {list(BOUNDARY_CONDITIONS)}  width: {WIDTHS}  method: {METHODS}",
         "결정타: boundary_50 smoothing 후 fall_prob 가 살아남(B)/붕괴(A)하느냐.\n"]
    for summ in summaries:
        stem = summ["session"]
        L.append(f"## {stem}")
        L.append(f"- original concat fall={summ['fall_prob_original_concat']:.3f} "
                 f"({summ['original_pred_class']})  center={summ['continuous_center_fall_prob']:.3f}  "
                 f"attn_b50_ratio={summ['original_attn_b50_ratio']:.3f}")
        L.append(f"- amp_jump_b50 orig={summ['amp_jump_b50_orig']:.3f}")
        # boundary_50_only 표
        L.append("- boundary_50_only fall_prob (method×width):")
        for method in METHODS:
            cells = []
            for w in WIDTHS:
                r = next(r for r in all_rows if r["session"] == stem and r["method"] == method
                         and r["boundary_condition"] == "boundary_50_only" and r["width"] == w)
                cells.append(f"w{w}={r['fall_prob_smoothed']:.3f}({'×' if not r['crosses_thr_0p1'] else '○'})")
            L.append(f"    {method}: " + "  ".join(cells))
        # both / 250 보조
        for cond in ("boundary_250_only", "both_boundaries"):
            cells = []
            for method in METHODS:
                for w in WIDTHS:
                    r = next(r for r in all_rows if r["session"] == stem and r["method"] == method
                             and r["boundary_condition"] == cond and r["width"] == w)
                    cells.append(f"{method[0]}w{w}={r['fall_prob_smoothed']:.3f}")
            L.append(f"- {cond}: " + "  ".join(cells))
        L.append(f"- boundary_50 small-width(3,5) mean fall={summ['boundary_50_small_width_mean_fall']:.3f}, "
                 f"drop_frac={summ['drop_frac_small_width']:.2f}, methods_agree={summ['methods_agree']}")
        L.append(f"- **verdict: {summ['verdict']}**\n")

    L.append("## 종합")
    for summ in summaries:
        L.append(f"- {summ['session']}: {summ['verdict']} "
                 f"(orig {summ['fall_prob_original_concat']:.3f} → b50 small-w mean {summ['boundary_50_small_width_mean_fall']:.3f})")
    return "\n".join(L) + "\n"


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT / "plots"
    if HAVE_MPL:
        plots_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[WARN] matplotlib 미사용 ({_MPL_ERR})")

    device = torch.device("cpu")
    if not CKPT.exists():
        raise FileNotFoundError(f"checkpoint 없음: {CKPT}")
    model, ck, classes = load_model(device)
    print(f"[ckpt] {CKPT_ID}: classes={classes} threshold={ck.get('threshold')}")

    all_rows, summaries = [], []
    for stem in TARGETS:
        rows, summ = analyze(stem, model, device)
        all_rows.extend(rows)
        summaries.append(summ)
        print(f"[an] {stem} orig={summ['fall_prob_original_concat']:.3f} "
              f"b50_small_mean={summ['boundary_50_small_width_mean_fall']:.3f} "
              f"drop={summ['drop_frac_small_width']:.2f} → {summ['verdict']}")
        if HAVE_MPL:
            plot_session(stem, rows, summ, plots_dir)

    # CSV
    header = ["session", "method", "boundary_condition", "width",
              "fall_prob_original_concat", "fall_prob_smoothed", "fall_prob_delta",
              "pred_class", "crosses_thr_0p1",
              "attention_boundary_50_ratio", "attention_boundary_250_ratio",
              "attention_peak_row", "sdp_row_energy_peak_row",
              "continuous_center_fall_prob", "distance_to_continuous_center",
              "amp_jump_b50_orig", "amp_jump_b250_orig", "amp_jump_b50_smoothed", "amp_jump_b250_smoothed"]

    def _r(x, n=5):
        if isinstance(x, bool):
            return x
        if isinstance(x, float):
            return round(x, n) if np.isfinite(x) else ""
        return "" if x is None else x

    with open(OUT / "walkb_smoothing_metrics.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for r in all_rows:
            w.writerow([_r(r.get(h)) for h in header])

    share = build_share(all_rows, summaries)
    (OUT / "walkb_smoothing_summary.md").write_text(share, encoding="utf-8")
    (OUT / "walkb_smoothing_summary.json").write_text(
        json.dumps({"meta": {"checkpoint_id": CKPT_ID, "threshold": THR,
                             "boundary_conditions": {k: v for k, v in BOUNDARY_CONDITIONS.items()},
                             "widths": WIDTHS, "methods": METHODS,
                             "note": "smoothing은 raw amplitude crop 레벨; 이후 정식 RPCA→SDP→모델 재실행."},
                    "summaries": summaries, "rows": all_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print("\n=== 산출 완료 ===")
    for f in ["walkb_smoothing_metrics.csv", "walkb_smoothing_summary.md", "walkb_smoothing_summary.json"]:
        print(f"  {OUT / f}")
    if HAVE_MPL:
        print(f"  {plots_dir}\\*.png")
    print("\n----- walkb_smoothing_summary.md -----")
    print(share)


if __name__ == "__main__":
    main()
