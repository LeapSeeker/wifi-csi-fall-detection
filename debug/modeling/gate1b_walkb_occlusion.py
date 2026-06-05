"""게이트 1-b 후속: WALK_B 3세션 메커니즘 확인 — row occlusion sensitivity (read-only).

질문: finetuned_baseline6 에서 WALK_B 의 concat_main fall_prob 상승이
  (a) splice boundary artifact 때문인가
  (b) continuous_center 의 stage2/stage3 beep 포함으로 인한 under-detection 인가.

대상 세션 (concat-only thr0.1 crossing 3건):
  E1_S01_A_FALL_WALK_B_T001  (concat 0.195 / center 0.019)
  E2_S02_A_FALL_WALK_B_T001  (concat 0.924 / center 0.015)
  E3_S03_A_FALL_WALK_B_T001  (concat 0.295 / center 0.046)

좌표:
  concat_main = clean[50:350] = orig[100:150]+[200:400]+[450:500]
    boundary local frames = 50, 250 → SDP boundary rows {3,4,5,23,24,25}
  continuous_center = orig[150:450]
    crop[0:50]=stage2 beep, crop[50:250]=fall/action, crop[250:300]=stage3 beep

수행:
  1. sparse frame energy curve (concat vs center)
  2. SDP heatmap / row energy (boundary rows vs transient rows)
  3. attention 위치 (peak row, boundary mass/ratio, transient mass/ratio) — finetuned_baseline6
  4. row occlusion sensitivity:
       z-scored SDP 입력 (1,28,20) 에서
       boundary rows {3,4,5,23,24,25} / transient rows {peak-1,peak,peak+1} 를
       0 masking (필수) + row-mean masking 으로 마스킹 후 fall_prob 재추론.
     delta_boundary  = fall_prob_original - fall_prob_boundary_occluded
     delta_transient = fall_prob_original - fall_prob_transient_occluded
     해석:
       boundary occlusion 영향 大 & transient 小 → (a) boundary artifact
       transient occlusion 영향 大 & boundary 小 → (b) transient 기반 (center under-detection)

제약: 동결 파일(pipeline/rpca/acf/sdp/학습·추론) 수정 금지. 새 학습 금지.
      데이터 read-only. 후처리 스크립트는 입력 tensor 만 조작(모델/전처리 코드 무수정).
      산출은 debug/modeling/diag_out/beep_concat_artifact/gate1b_occlusion/ 하위.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Windows cp949 콘솔에서 em-dash 등 유니코드 print 크래시 방지
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from model.preprocessing.loader import load_safesignal_csv          # noqa: E402
from model.preprocessing.resample import resample_to_100hz          # noqa: E402
from model.preprocessing.rpca import rpca_sparse, DEFAULT_MAX_ITER  # noqa: E402
from model.preprocessing.sdp import (                               # noqa: E402
    stacked_doppler_profile, SUB_W, SUB_STRIDE, W_T,
)
from model.preprocessing.acf import N_LAGS                          # noqa: E402
from model.preprocessing.pipeline import window_to_model_input      # noqa: E402
from model.pretrained.model import CNNGRUAttention                  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as _e:  # pragma: no cover
    HAVE_MPL = False
    _MPL_ERR = repr(_e)

OUT = ROOT / "debug/modeling/diag_out/beep_concat_artifact/gate1b_occlusion"
CLEANED = ROOT / "data/cleaned"
CKPT = ROOT / "model/finetune/checkpoints_compare6_cpu/best_operating.pt"
CKPT_ID = "finetuned_baseline6"

PRETRAINED6_CLASSES = ("fall", "walking", "sit_stand", "lying", "standing", "picking")
FALL_IDX = 0

TARGETS = [
    "E1_S01_A_FALL_WALK_B_T001",
    "E2_S02_A_FALL_WALK_B_T001",
    "E3_S03_A_FALL_WALK_B_T001",
]

BOUNDARIES = [50, 250]


def sdp_rows_for_frame(f, sub_w=SUB_W, stride=SUB_STRIDE, n_rows=W_T):
    lo = int(np.ceil((f - (sub_w - 1)) / stride))
    hi = int(np.floor(f / stride))
    return [i for i in range(max(0, lo), min(n_rows - 1, hi) + 1)]


BOUNDARY_ROWS = sorted(set(sum((sdp_rows_for_frame(b) for b in BOUNDARIES), [])))  # {3,4,5,23,24,25}
BOUNDARY_50_ROWS = sdp_rows_for_frame(50)    # {3,4,5}  — concat-local 50 = fall onset(orig200)/splice
BOUNDARY_250_ROWS = sdp_rows_for_frame(250)  # {23,24,25} — concat-local 250 = fall offset/splice
MIDBODY_ROWS = sdp_rows_for_frame(150)       # {13,14,15} — concat-local 150 = fall body 중심(통제군, non-boundary)

# 주의(교란): concat_main 의 boundary frame 50 = orig[200] = fall 본체 시작 = fall ONSET.
# 따라서 boundary rows 는 'splice 위치'이자 동시에 'fall onset/offset 위치'다.
# boundary occlusion 효과는 splice-artifact 와 legitimate-onset 을 분리하지 못한다.
# → 행 개수를 맞춘 boundary_50/boundary_250/transient_peak/midbody 통제 occlusion 으로 비교한다.


# ─── 전처리 (정식 경로 재현) ─────────────────────────────────────────────────
def zscore(sdp):
    return (sdp - sdp.mean()) / (sdp.std() + 1e-6)


def row_energy(sdp):
    return np.abs(sdp).mean(axis=1)  # (28,)


def frame_energy(mat):
    return np.abs(mat).mean(axis=1)  # (n_t,)


def preprocess_crop(crop):
    sparse = rpca_sparse(crop, max_iter=DEFAULT_MAX_ITER, tol=None)
    sdp_raw = stacked_doppler_profile(sparse, sub_w=SUB_W, stride=SUB_STRIDE, n_lags=N_LAGS)
    sdp_z = zscore(sdp_raw)
    return {"sparse": sparse, "sdp_raw": sdp_raw, "sdp_z": sdp_z, "model_input": sdp_z[None, ...]}


def build_crops(A550):
    clean = np.concatenate([A550[50:150], A550[200:400], A550[450:550]], axis=0)
    return {"concat_main": clean[50:350], "continuous_center": A550[150:450]}


# ─── 모델 ────────────────────────────────────────────────────────────────────
def load_model(device):
    ck = torch.load(CKPT, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    classes = list(ck.get("classes")) if isinstance(ck, dict) and ck.get("classes") else None
    if classes is not None and (len(classes) != 6 or classes[0] != "fall"):
        raise ValueError(f"classes 검증 실패: {classes}")
    if "classifier.1.weight" in sd and int(sd["classifier.1.weight"].shape[0]) != 6:
        raise ValueError("output dim != 6")
    m = CNNGRUAttention(n_classes=6).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, ck, classes


def infer(model, model_input, device, mask_rows=None, mode="zero", want_attention=False):
    """model_input (1,28,20). mask_rows 주면 해당 row 를 0(zero) 또는 per-lag row-mean 으로 대체.
    Returns (probs (6,), attn (28,) or None)."""
    x = np.array(model_input, dtype=np.float32, copy=True)  # (1,28,20)
    if mask_rows:
        if mode == "zero":
            x[0, mask_rows, :] = 0.0
        elif mode == "mean":
            col_mean = x[0].mean(axis=0)            # (20,) per-lag mean over rows
            x[0, mask_rows, :] = col_mean[None, :]
        else:
            raise ValueError(mode)
    xb = torch.as_tensor(x[None, ...], dtype=torch.float32).to(device)  # (1,1,28,20)
    with torch.no_grad():
        if want_attention:
            logits, weights = model(xb, return_attention=True)
            attn = weights.squeeze(0).cpu().numpy()
        else:
            logits = model(xb)
            attn = None
        p = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return p, attn


def att_block(attn, rows):
    a = np.asarray(attn, np.float64)
    mass = float(a[rows].sum()); tot = float(a.sum())
    return mass, (float(mass / tot) if tot else None)


# ─── 세션 찾기 ───────────────────────────────────────────────────────────────
def find_file(stem):
    hits = list(CLEANED.rglob(stem + ".csv"))
    if not hits:
        raise FileNotFoundError(f"세션 파일 없음: {stem}")
    return hits[0]


# ─── 분석 ────────────────────────────────────────────────────────────────────
def analyze(stem, model, device):
    path = find_file(stem)
    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    A = res.amplitude
    n = int(res.resampled_count)
    A550 = A[:550]
    crops = build_crops(A550)

    rec = {"session": stem, "n_frames": n, "crops": {}}
    arrays = {}
    for cname, crop in crops.items():
        pp = preprocess_crop(crop)
        mi = pp["model_input"]  # (1,28,20)

        # 정식 경로 일치 1회 확인
        consistent = bool(np.allclose(mi[0], window_to_model_input(crop), atol=1e-4))

        re_z = row_energy(pp["sdp_z"])              # (28,)
        fe = frame_energy(pp["sparse"])             # (300,)
        peak_row = int(re_z.argmax())
        transient_rows = [r for r in (peak_row - 1, peak_row, peak_row + 1) if 0 <= r < W_T]

        # 원본 추론 + attention
        p_orig, attn = infer(model, mi, device, want_attention=True)
        fall_orig = float(p_orig[FALL_IDX])

        # occlusion row sets (transient 제외 모두 3-row 로 개수 일치한 통제 비교)
        row_sets = {
            "boundary_all": BOUNDARY_ROWS,        # 6 rows (사양 요구)
            "boundary_50": BOUNDARY_50_ROWS,      # 3 rows — onset/splice
            "boundary_250": BOUNDARY_250_ROWS,    # 3 rows — offset/splice
            "transient_peak": transient_rows,     # ~3 rows — SDP energy peak (mid-body)
            "midbody_ctrl": MIDBODY_ROWS,         # 3 rows — fall 중심 통제군
        }
        occ = {}      # set -> {zero, mean, delta_zero, delta_mean}
        att_ratio = {}
        for sname, rows in row_sets.items():
            _, ratio = att_block(attn, rows)
            att_ratio[sname] = ratio
            for mode in ("zero", "mean"):
                p_occ, _ = infer(model, mi, device, mask_rows=rows, mode=mode)
                fp = float(p_occ[FALL_IDX])
                occ.setdefault(sname, {})[mode] = fp
                occ[sname][f"delta_{mode}"] = fall_orig - fp

        rec["crops"][cname] = {
            "consistent_with_official": consistent,
            "fall_prob_original": fall_orig,
            "pred_class": PRETRAINED6_CLASSES[int(p_orig.argmax())],
            "sparse_peak_frame": int(fe.argmax()),
            "sdp_peak_row": peak_row,
            "transient_rows": transient_rows,
            "transient_rows_overlap_boundary": sorted(set(transient_rows) & set(BOUNDARY_ROWS)),
            "attention_peak_row": int(np.argmax(attn)),
            "attention_peak_value": float(np.max(attn)),
            "attention_ratio": att_ratio,   # set 별 attention mass 비율
            "occlusion": occ,               # set 별 fall_prob + delta (zero/mean)
            # 사양 호환 단축 키 (boundary_all vs transient_peak, zero 기준)
            "delta_boundary": occ["boundary_all"]["delta_zero"],
            "delta_transient": occ["transient_peak"]["delta_zero"],
            "delta_transient_minus_boundary": occ["transient_peak"]["delta_zero"] - occ["boundary_all"]["delta_zero"],
        }
        arrays[cname] = {"sparse_frame_energy": fe, "sdp_z": pp["sdp_z"],
                         "row_energy": re_z, "attn": attn,
                         "peak_row": peak_row, "transient_rows": transient_rows}

    # ── 메커니즘 판정 (concat_main 기준, 3-row 통제 set 비교) ─────────────────
    cm = rec["crops"]["concat_main"]
    occ = cm["occlusion"]
    cand = {k: occ[k]["delta_zero"] for k in ("boundary_50", "boundary_250", "transient_peak", "midbody_ctrl")}
    primary = max(cand, key=lambda k: cand[k])  # 제거 시 fall 가장 크게 떨어지는 set
    if cand[primary] < 0.02:
        verdict = "inconclusive_small_effect"
    elif primary in ("boundary_50", "boundary_250"):
        verdict = "a_boundary_onset_driven"   # onset/offset(=splice) row 가 주도 → (a)-leaning
    else:
        verdict = "b_transient_driven"        # 본체 transient row 가 주도 → (b)-leaning
    rec["concat_mechanism_verdict"] = verdict
    rec["concat_primary_driver_row_set"] = primary
    rec["concat_delta_by_set_zero"] = cand
    ar = cm["attention_ratio"]
    rec["concat_attention_ratio_boundary_all_vs_transient"] = [ar["boundary_all"], ar["transient_peak"]]
    rec["_arrays"] = arrays
    return rec


# ─── 그림 ────────────────────────────────────────────────────────────────────
def plot_session(rec, plots_dir):
    if not HAVE_MPL:
        return
    stem = rec["session"]
    arr = rec["_arrays"]
    crops = ["concat_main", "continuous_center"]

    # 1) sparse frame energy
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharey=True)
    for ax, c in zip(axes, crops):
        fe = arr[c]["sparse_frame_energy"]
        ax.plot(np.arange(len(fe)), fe, lw=0.9)
        if c == "concat_main":
            for b in BOUNDARIES:
                ax.axvline(b, color="C3", ls="--", lw=1.0)
            ax.set_title(f"{c} (boundary 50/250)", fontsize=9)
        else:
            ax.axvspan(0, 50, color="0.85"); ax.axvspan(250, 300, color="0.85")
            ax.set_title(f"{c} (beep 0-50, 250-300 음영)", fontsize=9)
        ax.set_xlabel("crop-local frame")
    axes[0].set_ylabel("mean|sparse|")
    fig.suptitle(f"{stem} | sparse frame energy", fontsize=10)
    fig.tight_layout(); fig.savefig(plots_dir / f"{stem}__sparse_energy.png", dpi=110); plt.close(fig)

    # 2) SDP row energy
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharey=True)
    for ax, c in zip(axes, crops):
        re = arr[c]["row_energy"]; pk = arr[c]["peak_row"]
        ax.plot(np.arange(len(re)), re, marker="o", ms=3, lw=0.9)
        for i in BOUNDARY_ROWS:
            ax.axvline(i, color="C3", ls="--", lw=0.7)
        ax.axvline(pk, color="C2", ls="-", lw=1.4, label=f"peak row {pk}")
        ax.set_title(c, fontsize=9); ax.set_xlabel("SDP row"); ax.legend(fontsize=7)
    axes[0].set_ylabel("mean|SDP_z[row,:]|")
    fig.suptitle(f"{stem} | SDP row energy (red=boundary rows, green=transient peak)", fontsize=10)
    fig.tight_layout(); fig.savefig(plots_dir / f"{stem}__sdp_row_energy.png", dpi=110); plt.close(fig)

    # 3) SDP heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    vmax = max(np.abs(arr[c]["sdp_z"]).max() for c in crops)
    for ax, c in zip(axes, crops):
        im = ax.imshow(arr[c]["sdp_z"], aspect="auto", origin="lower", cmap="magma", vmin=-vmax, vmax=vmax)
        for i in BOUNDARY_ROWS:
            ax.axhline(i, color="cyan", ls=":", lw=0.5)
        ax.set_title(c, fontsize=9); ax.set_xlabel("lag"); ax.set_ylabel("row")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.04)
    fig.suptitle(f"{stem} | SDP(z) heatmap", fontsize=10)
    fig.savefig(plots_dir / f"{stem}__sdp_heatmap.png", dpi=110); plt.close(fig)

    # 4) attention
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for c in crops:
        a = arr[c]["attn"]
        ax.plot(np.arange(len(a)), a, marker="o", ms=2.5, lw=0.9, label=c)
    for i in BOUNDARY_ROWS:
        ax.axvline(i, color="0.6", ls=":", lw=0.7)
    ax.set_title(f"{stem} | attention over rows ({CKPT_ID})  boundary={BOUNDARY_ROWS}", fontsize=9)
    ax.set_xlabel("SDP row"); ax.set_ylabel("attention"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plots_dir / f"{stem}__attention.png", dpi=110); plt.close(fig)


# ─── share / verdict 텍스트 ──────────────────────────────────────────────────
def build_share(records):
    L = ["# 게이트 1-b WALK_B 메커니즘 확인 — row occlusion sensitivity\n",
         f"checkpoint: {CKPT_ID}",
         f"occlusion row sets (zero masking): boundary_50={BOUNDARY_50_ROWS} boundary_250={BOUNDARY_250_ROWS} "
         f"transient_peak={{peak±1}} midbody_ctrl={MIDBODY_ROWS}",
         "Δ = fall_prob_original - fall_prob_occluded. 클수록 그 row set 제거가 fall 을 크게 떨어뜨림(=의존 큼).",
         "교란: concat boundary frame 50/250 = fall 본체 시작/끝(onset/offset)이자 splice 위치. "
         "boundary_50/250 의존은 splice-artifact 와 legitimate-onset 을 분리하지 못함.\n"]
    for r in records:
        cm = r["crops"]["concat_main"]; ct = r["crops"]["continuous_center"]
        co = cm["occlusion"]; ar = cm["attention_ratio"]
        L.append(f"## {r['session']}")
        L.append(f"- fall_prob: concat={cm['fall_prob_original']:.3f} ({cm['pred_class']})  "
                 f"center={ct['fall_prob_original']:.3f} ({ct['pred_class']})")
        L.append(f"- concat sdp_peak_row={cm['sdp_peak_row']}, attn_peak_row={cm['attention_peak_row']}, "
                 f"sparse_peak_frame={cm['sparse_peak_frame']}")
        L.append(f"- concat attention ratio: boundary_all={ar['boundary_all']:.3f} "
                 f"boundary_50={ar['boundary_50']:.3f} boundary_250={ar['boundary_250']:.3f} "
                 f"transient={ar['transient_peak']:.3f}")
        L.append(f"- concat occlusion Δ(zero): boundary_50={co['boundary_50']['delta_zero']:+.3f}  "
                 f"boundary_250={co['boundary_250']['delta_zero']:+.3f}  "
                 f"transient_peak={co['transient_peak']['delta_zero']:+.3f}  "
                 f"midbody_ctrl={co['midbody_ctrl']['delta_zero']:+.3f}")
        L.append(f"- **primary driver = {r['concat_primary_driver_row_set']} "
                 f"→ verdict: {r['concat_mechanism_verdict']}**\n")

    verdicts = [r["concat_mechanism_verdict"] for r in records]
    na = sum(v == "a_boundary_onset_driven" for v in verdicts)
    nb = sum(v == "b_transient_driven" for v in verdicts)
    ni = sum(v == "inconclusive_small_effect" for v in verdicts)
    L.append("## 종합")
    L.append(f"- a_boundary_onset_driven={na}, b_transient_driven={nb}, inconclusive={ni} (of {len(records)})")
    L.append("- 주의: a_boundary_onset_driven 은 'splice artifact' 와 'legitimate fall onset' 을 분리 못 함 "
             "(둘 다 frame 50/250 에 공존). midbody_ctrl 대비 boundary 우세 정도로 해석 보조.")
    return "\n".join(L) + "\n"


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT / "plots"
    if HAVE_MPL:
        plots_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[WARN] matplotlib 미사용 ({_MPL_ERR}) → PNG 생략")

    device = torch.device("cpu")
    if not CKPT.exists():
        raise FileNotFoundError(f"checkpoint 없음: {CKPT}")
    model, ck, classes = load_model(device)
    print(f"[ckpt] {CKPT_ID}: classes={classes} threshold={ck.get('threshold')} device={device}")

    records = []
    for stem in TARGETS:
        rec = analyze(stem, model, device)
        cm = rec["crops"]["concat_main"]
        print(f"[an] {stem} concat={cm['fall_prob_original']:.3f} "
              f"Δbnd={cm['delta_boundary']:+.3f} Δtrans={cm['delta_transient']:+.3f} "
              f"→ {rec['concat_mechanism_verdict']}")
        if HAVE_MPL:
            plot_session(rec, plots_dir)
        rec.pop("_arrays", None)
        records.append(rec)

    # CSV (per session × crop). set 별 delta(zero) 와 attention ratio 를 평탄화.
    SETS = ["boundary_all", "boundary_50", "boundary_250", "transient_peak", "midbody_ctrl"]
    header = (["session", "crop", "fall_prob_original", "pred_class",
               "sparse_peak_frame", "sdp_peak_row", "transient_rows", "attention_peak_row"]
              + [f"att_ratio__{s}" for s in SETS]
              + [f"fallprob_occ_zero__{s}" for s in SETS]
              + [f"delta_zero__{s}" for s in SETS]
              + [f"delta_mean__{s}" for s in SETS])

    def _r(x, n=5):
        return round(x, n) if isinstance(x, (int, float)) and x is not None else ("" if x is None else x)

    with open(OUT / "walkb_occlusion_metrics.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for r in records:
            for c in ("concat_main", "continuous_center"):
                d = r["crops"][c]
                row = [r["session"], c, _r(d["fall_prob_original"]), d["pred_class"],
                       d["sparse_peak_frame"], d["sdp_peak_row"],
                       "|".join(map(str, d["transient_rows"])), d["attention_peak_row"]]
                row += [_r(d["attention_ratio"][s], 4) for s in SETS]
                row += [_r(d["occlusion"][s]["zero"]) for s in SETS]
                row += [_r(d["occlusion"][s]["delta_zero"]) for s in SETS]
                row += [_r(d["occlusion"][s]["delta_mean"]) for s in SETS]
                w.writerow(row)

    share = build_share(records)
    (OUT / "walkb_occlusion_summary.md").write_text(share, encoding="utf-8")
    (OUT / "walkb_occlusion_summary.json").write_text(
        json.dumps({"meta": {"checkpoint_id": CKPT_ID, "ckpt": str(CKPT),
                             "boundary_rows": BOUNDARY_ROWS,
                             "threshold": ck.get("threshold"), "classes": classes},
                    "sessions": records}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 산출 완료 ===")
    for f in ["walkb_occlusion_metrics.csv", "walkb_occlusion_summary.md", "walkb_occlusion_summary.json"]:
        print(f"  {OUT / f}")
    if HAVE_MPL:
        print(f"  {plots_dir}\\*.png")
    print("\n----- walkb_occlusion_summary.md -----")
    print(share)


if __name__ == "__main__":
    main()
