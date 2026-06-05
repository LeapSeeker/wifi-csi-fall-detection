"""게이트 1: beep 제거 concat 접합부 artifact sanity 진단 (read-only).

D-031 event-centered 구현 전 선행 게이트. clean400(beep 제거 concat)의
RPCA→ACF→SDP 접합부가 비정상 에너지/단서를 만드는지 두 층으로 진단한다.

  (A) artifact-only : concat crop 내부 접합부(crop-local 50, 250)가 주변 대비 튀는지.
  (B) fallback 비교 : concat_main vs 원본 continuous crop 후보(beep 미제거)의
                       SDP / 모델 확률 차이. *동일 내용 1:1 비교가 아니라 대체 후보 비교*.

좌표계 (clean400 — D-031 항목 1 확정)
  resampled original = amp[:550]
  clean = original[50:150] + original[200:400] + original[450:550]   # 100+200+100 = 400
  concat_main = clean[50:350]                                         # 300f (모델 정합)
              = original[100:150] + original[200:400] + original[450:500]
  concat_main 접합부 (crop-local):
    frame 50  = original[149] -> original[200] 접합 (clean abs 100)
    frame 250 = original[399] -> original[450] 접합 (clean abs 300)

continuous fallback 후보 (각 300f, beep 미제거 — 공정한 1:1 대응 아님)
  continuous_pre_fall  = original[100:400]
  continuous_center    = original[150:450]
  continuous_fall_post = original[200:500]

전처리 경로 (각 crop, 정식 함수와 동일)
  crop(300,n_sc) -> rpca_sparse(max_iter=200, tol=None)
  -> stacked_doppler_profile(sub_w=30, stride=10, n_lags=20) -> (28,20)
  -> global z-score -> model input (1,28,20)

판정은 진규/Codex 검토. 본 스크립트는 수치/그림과 해석 보조 플래그만 산출한다.

제약: 동결 파일(pipeline/rpca/acf/sdp/학습·추론) 및 데이터 read-only.
      산출물은 debug/modeling/diag_out/beep_concat_artifact/ 하위에만 기록.

사용법:
  python debug/modeling/gate1_beep_concat_artifact.py
  python debug/modeling/gate1_beep_concat_artifact.py --ckpt model/finetune/checkpoints_xxx/best.pt
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

from model.preprocessing.loader import load_safesignal_csv          # noqa: E402
from model.preprocessing.resample import resample_to_100hz          # noqa: E402
from model.preprocessing.rpca import rpca_sparse, DEFAULT_MAX_ITER  # noqa: E402
from model.preprocessing.sdp import (                               # noqa: E402
    stacked_doppler_profile, SUB_W, SUB_STRIDE, W_T,
)
from model.preprocessing.acf import N_LAGS                          # noqa: E402

# matplotlib 은 그림 산출에만 필요. 없으면 수치만 산출하고 경고.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as _e:  # pragma: no cover
    HAVE_MPL = False
    _MPL_ERR = repr(_e)

OUT = ROOT / "debug/modeling/diag_out/beep_concat_artifact"
CLEANED = ROOT / "data/cleaned"

# subtype 6종 각 2개. (직전 beep_concat_artifact / beep_incrop 진단과 동일 풀)
SUBTYPES = ["FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B", "FALL_WALK_F", "FALL_WALK_B"]
PER_SUBTYPE = 2
MIN_FRAMES = 550  # resampled_count 하한. 미달 세션은 같은 subtype 다른 trial로 대체.

# 접합부 (crop-local frame index of first frame AFTER splice)
BOUNDARIES = [50, 250]

# continuous fallback 후보: 원본 인덱스 [lo, hi) (각 300f)
CONTINUOUS = {
    "continuous_pre_fall": (100, 400),
    "continuous_center": (150, 450),
    "continuous_fall_post": (200, 500),
}
ALL_CROPS = ["concat_main"] + list(CONTINUOUS.keys())


# ─── SDP bin ↔ frame 매핑 (기존 진단 스크립트와 동일 정의) ─────────────────────
def sdp_rows_for_frame(f, sub_w=SUB_W, stride=SUB_STRIDE, n_rows=W_T):
    """frame f 를 포함하는 SDP row 집합. row i 는 frame [stride*i, stride*i+sub_w).
    i in [ceil((f-(sub_w-1))/stride), floor(f/stride)] ∩ [0, n_rows-1].
    b=50 -> {3,4,5}, b=250 -> {23,24,25}."""
    lo = int(np.ceil((f - (sub_w - 1)) / stride))
    hi = int(np.floor(f / stride))
    return [i for i in range(max(0, lo), min(n_rows - 1, hi) + 1)]


# ─── 전처리 경로 헬퍼 (정식 window_to_model_input 과 동일 경로 재현) ──────────
def zscore(sdp):
    return (sdp - sdp.mean()) / (sdp.std() + 1e-6)


def row_energy(sdp):
    """SDP row energy = mean(abs(sdp[row,:]))  → (28,)."""
    return np.abs(sdp).mean(axis=1)


def frame_energy(mat):
    """time-frame energy = mean(abs(mat[t,:])) (서브캐리어 평균)  → (n_t,)."""
    return np.abs(mat).mean(axis=1)


def preprocess_crop(crop):
    """crop(300,n_sc) → dict(sparse, sdp_raw, sdp_z, model_input(1,28,20)).

    정식 window_to_model_input 내부와 동일:
      sparse = rpca_sparse(crop, max_iter=200, tol=None)
      sdp_raw = stacked_doppler_profile(sparse)
      sdp_z = (sdp_raw - mean)/(std+1e-6)
      model_input = sdp_z[None, ...]
    raw SDP(pre-zscore)와 z-scored SDP를 모두 보존한다.
    """
    sparse = rpca_sparse(crop, max_iter=DEFAULT_MAX_ITER, tol=None)
    sdp_raw = stacked_doppler_profile(sparse, sub_w=SUB_W, stride=SUB_STRIDE, n_lags=N_LAGS)
    sdp_z = zscore(sdp_raw)
    return {
        "sparse": sparse,
        "sdp_raw": sdp_raw,
        "sdp_z": sdp_z,
        "model_input": sdp_z[None, ...],  # (1,28,20)
    }


# ─── 비교 지표 ────────────────────────────────────────────────────────────────
def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def mad(a, b):
    """mean absolute difference."""
    return float(np.mean(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def pearson(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ─── 모델 (선택 옵션 --ckpt) ─────────────────────────────────────────────────
def load_pretrained6_model(ckpt_path, device):
    """6-class pretrained6 계열 checkpoint 로드 + 검증.

    검증: output dim == 6, classes metadata 있으면 classes[0] == 'fall'.
    Returns (model, info) — info 는 summary.json 기록용 메타.
    """
    import torch
    from model.pretrained.model import CNNGRUAttention

    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    # output dim: classifier 마지막 Linear weight 행 수
    out_w = None
    for key in ("classifier.1.weight", "classifier.weight"):
        if key in sd:
            out_w = sd[key]
            break
    if out_w is None:
        cand = [k for k in sd if k.endswith("weight") and sd[k].ndim == 2]
        out_w = sd[cand[-1]] if cand else None
    out_dim = int(out_w.shape[0]) if out_w is not None else None
    if out_dim is not None and out_dim != 6:
        raise ValueError(f"checkpoint output dim != 6 (got {out_dim}). pretrained6 계열만 허용.")

    classes = ck.get("classes") if isinstance(ck, dict) else None
    if classes is not None:
        classes = list(classes)
        if len(classes) != 6:
            raise ValueError(f"checkpoint classes 길이 != 6 (got {len(classes)}): {classes}")
        if classes[0] != "fall":
            raise ValueError(f"checkpoint classes[0] != 'fall' (got {classes[0]!r}).")

    model = CNNGRUAttention(n_classes=6).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    # dummy forward 로 출력 차원 재확인
    with torch.no_grad():
        o = model(torch.zeros(1, 1, W_T, N_LAGS, device=device))
    if int(o.shape[1]) != 6:
        raise ValueError(f"forward output dim != 6 (got {int(o.shape[1])}).")

    info = {
        "ckpt": str(ckpt_path),
        "output_dim": out_dim if out_dim is not None else 6,
        "classes": classes,
        "fall_index": 0,
        "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
        "saved_threshold": ck.get("threshold") if isinstance(ck, dict) else None,
    }
    return model, info


def infer_softmax(model, model_input, device):
    """model_input (1,28,20) → softmax probs (6,)."""
    import torch
    xb = torch.as_tensor(model_input[None, ...], dtype=torch.float32).to(device)  # (1,1,28,20)
    with torch.no_grad():
        p = torch.softmax(model(xb), dim=1).cpu().numpy()[0]
    return p


# ─── crop 구성 ───────────────────────────────────────────────────────────────
def build_clean400(A550):
    return np.concatenate([A550[50:150], A550[200:400], A550[450:550]], axis=0)  # (400, dim)


def build_concat_main(A550):
    return build_clean400(A550)[50:350]  # (300, dim)


def build_crops(A550):
    crops = {"concat_main": build_concat_main(A550)}
    for name, (lo, hi) in CONTINUOUS.items():
        crops[name] = A550[lo:hi]
    return crops


# ─── 세션 선택 (resampled_count >= 550) ──────────────────────────────────────
def select_sessions():
    """subtype별 PER_SUBTYPE개 (resampled_count>=550). 미달은 다른 trial로 대체.

    Returns (selected, records)
      selected : [{subtype, path, filename, resampled_count, is_substitute}]
      records  : selected_sessions.csv 기록용 전체 결정 로그.
    """
    selected = []
    records = []
    for st in SUBTYPES:
        files = sorted(CLEANED.rglob(f"*{st}*_T*.csv"))
        got = 0
        saw_short = False
        if not files:
            records.append({"subtype": st, "filename": "", "resampled_count": "",
                            "decision": "NO_FILE", "is_substitute": "",
                            "reason": "subtype 매칭 파일 없음"})
            print(f"[WARN] {st}: 매칭 파일 없음")
            continue
        for f in files:
            try:
                raw = load_safesignal_csv(f, rx="both")
                res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
                n = int(res.resampled_count)
            except Exception as e:  # 로드/리샘플 실패도 기록 후 다음 trial
                records.append({"subtype": st, "filename": f.name, "resampled_count": "",
                                "decision": "LOAD_ERROR", "is_substitute": "",
                                "reason": repr(e)[:120]})
                print(f"[skip] {st} {f.name}: LOAD_ERROR {e}")
                continue
            if n < MIN_FRAMES:
                saw_short = True
                records.append({"subtype": st, "filename": f.name, "resampled_count": n,
                                "decision": "EXCLUDED_SHORT", "is_substitute": "",
                                "reason": f"resampled_count<{MIN_FRAMES}"})
                print(f"[skip] {st} {f.name}: EXCLUDED_SHORT (n={n})")
                continue
            if got >= PER_SUBTYPE:
                records.append({"subtype": st, "filename": f.name, "resampled_count": n,
                                "decision": "NOT_NEEDED", "is_substitute": "",
                                "reason": f"이미 {PER_SUBTYPE}개 확보"})
                continue
            is_sub = saw_short  # 앞선 trial 제외 후 채택 → 대체
            selected.append({"subtype": st, "path": f, "filename": f.name,
                             "resampled_count": n, "is_substitute": is_sub})
            records.append({"subtype": st, "filename": f.name, "resampled_count": n,
                            "decision": "SELECTED", "is_substitute": is_sub,
                            "reason": "substitute" if is_sub else "primary"})
            print(f"[OK]   {st} {f.name} n={n}{' (substitute)' if is_sub else ''}")
            got += 1
        if got < PER_SUBTYPE:
            print(f"[WARN] {st}: 정상길이(>= {MIN_FRAMES}) {got}/{PER_SUBTYPE}개만 확보")
    return selected, records


# ─── 세션 분석 ───────────────────────────────────────────────────────────────
def analyze_session(sess, model, device):
    raw = load_safesignal_csv(sess["path"], rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    A = res.amplitude
    n = int(res.resampled_count)
    dim = int(A.shape[1])
    A550 = A[:550]

    crops = build_crops(A550)
    pp = {name: preprocess_crop(crop) for name, crop in crops.items()}

    # ── per-crop 스칼라 지표 ─────────────────────────────────────────────────
    per_crop = {}
    for name in ALL_CROPS:
        d = pp[name]
        re_raw = row_energy(d["sdp_raw"])
        re_z = row_energy(d["sdp_z"])
        fe = frame_energy(d["sparse"])
        per_crop[name] = {
            "crop_range": ([100, 150, 200, 400, 450, 500] if name == "concat_main"
                           else list(CONTINUOUS[name])),
            "sdp_raw_peak_row": int(re_raw.argmax()),
            "sdp_z_peak_row": int(re_z.argmax()),
            "sdp_raw_energy_mean": float(re_raw.mean()),
            "sdp_z_energy_mean": float(re_z.mean()),
            "sparse_energy_peak_frame": int(fe.argmax()),
            "sparse_energy_mean": float(fe.mean()),
        }

    # ── pairwise: concat_main vs each continuous ─────────────────────────────
    cm = pp["concat_main"]
    cm_re_raw = row_energy(cm["sdp_raw"])
    cm_fe = frame_energy(cm["sparse"])
    pairwise = {}
    for name in CONTINUOUS:
        d = pp[name]
        re_raw = row_energy(d["sdp_raw"])
        fe = frame_energy(d["sparse"])
        pairwise[name] = {
            "raw_sdp_cosine": cosine(cm["sdp_raw"], d["sdp_raw"]),
            "raw_sdp_mad": mad(cm["sdp_raw"], d["sdp_raw"]),
            "z_sdp_cosine": cosine(cm["sdp_z"], d["sdp_z"]),
            "z_sdp_mad": mad(cm["sdp_z"], d["sdp_z"]),
            "sdp_energy_curve_corr": pearson(cm_re_raw, re_raw),
            "sdp_peak_row_diff": int(abs(int(cm_re_raw.argmax()) - int(re_raw.argmax()))),
            "sparse_peak_frame_diff": int(abs(int(cm_fe.argmax()) - int(fe.argmax()))),
        }

    # ── 접합부 artifact 직접 진단 (concat_main 전용) ─────────────────────────
    crop = crops["concat_main"]
    adj = np.abs(np.diff(crop, axis=0)).mean(axis=1)        # (299,) adj[i]=mean|crop[i+1]-crop[i]|
    bnd_adj_idx = [b - 1 for b in BOUNDARIES]               # crop[b]-crop[b-1] == adj[b-1]
    nonbnd_adj = np.delete(adj, bnd_adj_idx)
    adj_nonbnd_median = float(np.median(nonbnd_adj))

    fe_cm = cm_fe                                            # sparse frame energy (300,)
    nbr = set()
    for b in BOUNDARIES:
        nbr.update([b - 1, b, b + 1])
    nonbnd_fe = np.array([fe_cm[t] for t in range(len(fe_cm)) if t not in nbr])
    fe_nonbnd_median = float(np.median(nonbnd_fe))

    re_z_cm = row_energy(cm["sdp_z"])                        # (28,)
    all_bnd_rows = sorted(set(sum((sdp_rows_for_frame(b) for b in BOUNDARIES), [])))
    nonbnd_rows = [i for i in range(W_T) if i not in all_bnd_rows]
    rows_nonbnd_median = float(np.median(re_z_cm[nonbnd_rows]))

    boundary = {}
    for b in BOUNDARIES:
        amp_jump = float(adj[b - 1])
        sparse_be = float(fe_cm[b])
        rows_b = sdp_rows_for_frame(b)
        rows_b_median = float(np.median(re_z_cm[rows_b]))
        boundary[b] = {
            "amp_jump": amp_jump,
            "amp_nonbnd_median": adj_nonbnd_median,
            "amp_jump_ratio": float(amp_jump / adj_nonbnd_median) if adj_nonbnd_median else None,
            "sparse_be_prev": float(fe_cm[b - 1]),
            "sparse_be": sparse_be,
            "sparse_be_next": float(fe_cm[b + 1]),
            "sparse_nonbnd_median": fe_nonbnd_median,
            "sparse_be_ratio": float(sparse_be / fe_nonbnd_median) if fe_nonbnd_median else None,
            "sdp_boundary_rows": rows_b,
            "sdp_boundary_rows_energy_median": rows_b_median,
            "sdp_nonbnd_rows_median": rows_nonbnd_median,
            "sdp_row_ratio": float(rows_b_median / rows_nonbnd_median) if rows_nonbnd_median else None,
        }

    # ── 모델 확률 (선택) ─────────────────────────────────────────────────────
    probs = {}
    model_block = None
    if model is not None:
        for name in ALL_CROPS:
            probs[name] = infer_softmax(model, pp[name]["model_input"], device)
        cm_p = probs["concat_main"]
        model_block = {
            "fall_prob": {name: float(probs[name][0]) for name in ALL_CROPS},
            "argmax_idx": {name: int(probs[name].argmax()) for name in ALL_CROPS},
            "vs_continuous": {
                name: {
                    "fall_prob_diff": float(cm_p[0] - probs[name][0]),   # concat - continuous
                    "softmax_l1_diff": float(np.abs(cm_p - probs[name]).sum()),
                }
                for name in CONTINUOUS
            },
        }

    return {
        "subtype": sess["subtype"],
        "file": sess["filename"],
        "n_frames": n,
        "dim": dim,
        "is_substitute": bool(sess["is_substitute"]),
        "per_crop": per_crop,
        "pairwise": pairwise,
        "boundary": boundary,
        "model": model_block,
        "_arrays": {  # 그림/npz 용 (summary.json 에는 미포함)
            "sdp_z": {name: pp[name]["sdp_z"] for name in ALL_CROPS},
            "sdp_raw": {name: pp[name]["sdp_raw"] for name in ALL_CROPS},
            "sparse_frame_energy_concat": fe_cm,
            "sdp_row_energy_concat_z": re_z_cm,
            "probs": probs,
        },
    }


# ─── 그림 ────────────────────────────────────────────────────────────────────
def plot_session(rec, plots_dir, have_model):
    if not HAVE_MPL:
        return
    arr = rec["_arrays"]
    stem = f"{rec['subtype']}__{Path(rec['file']).stem}"

    # 1) sparse frame energy curve + boundary markers
    fe = arr["sparse_frame_energy_concat"]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(len(fe)), fe, lw=0.9, color="C0")
    for b in BOUNDARIES:
        ax.axvline(b, color="C3", ls="--", lw=1.0, label=f"boundary {b}" if b == BOUNDARIES[0] else None)
    ax.set_title(f"{stem} | concat_main sparse frame energy")
    ax.set_xlabel("crop-local frame"); ax.set_ylabel("mean|sparse|")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(plots_dir / f"{stem}__sparse_energy.png", dpi=110); plt.close(fig)

    # 2) SDP row energy curve + boundary row markers
    re = arr["sdp_row_energy_concat_z"]
    bnd_rows = sorted(set(sum((sdp_rows_for_frame(b) for b in BOUNDARIES), [])))
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(len(re)), re, marker="o", ms=3, lw=0.9, color="C0")
    for i in bnd_rows:
        ax.axvline(i, color="C3", ls="--", lw=0.8)
    ax.set_title(f"{stem} | concat_main SDP(z) row energy  (boundary rows {bnd_rows})")
    ax.set_xlabel("SDP row"); ax.set_ylabel("mean|SDP_z[row,:]|")
    fig.tight_layout(); fig.savefig(plots_dir / f"{stem}__sdp_row_energy.png", dpi=110); plt.close(fig)

    # 3) concat vs continuous SDP(z) heatmaps
    fig, axes = plt.subplots(1, len(ALL_CROPS), figsize=(3.2 * len(ALL_CROPS), 3.4))
    vmax = max(np.abs(arr["sdp_z"][n]).max() for n in ALL_CROPS)
    for ax, name in zip(axes, ALL_CROPS):
        im = ax.imshow(arr["sdp_z"][name], aspect="auto", origin="lower",
                       cmap="magma", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("lag"); ax.set_ylabel("row")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025)
    fig.suptitle(f"{stem} | SDP(z) heatmaps", fontsize=10)
    fig.savefig(plots_dir / f"{stem}__sdp_heatmaps.png", dpi=110); plt.close(fig)

    # 4) fall probability bar plot (ckpt 사용 시)
    if have_model and rec["model"] is not None:
        fp = rec["model"]["fall_prob"]
        fig, ax = plt.subplots(figsize=(6, 3))
        names = ALL_CROPS
        ax.bar(range(len(names)), [fp[n] for n in names], color="C0")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_ylabel("fall_prob")
        ax.set_title(f"{stem} | fall probability")
        fig.tight_layout(); fig.savefig(plots_dir / f"{stem}__fall_prob.png", dpi=110); plt.close(fig)


# ─── 집계 ────────────────────────────────────────────────────────────────────
def stats(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "mean": float(a.mean()),
    }


def aggregate(records, have_model):
    """전체 및 분포 집계. records 는 analyze_session 결과 리스트."""
    def collect_boundary(metric):
        return [records_i["boundary"][b][metric] for records_i in records for b in BOUNDARIES]

    agg = {
        "boundary_artifact_ratio": {
            "amp_jump_ratio": stats(collect_boundary("amp_jump_ratio")),
            "sparse_be_ratio": stats(collect_boundary("sparse_be_ratio")),
            "sdp_row_ratio": stats(collect_boundary("sdp_row_ratio")),
        },
        "pairwise_by_candidate": {},
    }
    for name in CONTINUOUS:
        agg["pairwise_by_candidate"][name] = {
            "raw_sdp_cosine": stats([r["pairwise"][name]["raw_sdp_cosine"] for r in records]),
            "z_sdp_cosine": stats([r["pairwise"][name]["z_sdp_cosine"] for r in records]),
            "raw_sdp_mad": stats([r["pairwise"][name]["raw_sdp_mad"] for r in records]),
            "z_sdp_mad": stats([r["pairwise"][name]["z_sdp_mad"] for r in records]),
            "sdp_energy_curve_corr": stats([r["pairwise"][name]["sdp_energy_curve_corr"] for r in records]),
            "sdp_peak_row_diff": stats([r["pairwise"][name]["sdp_peak_row_diff"] for r in records]),
            "sparse_peak_frame_diff": stats([r["pairwise"][name]["sparse_peak_frame_diff"] for r in records]),
        }
    if have_model:
        agg["model_fall_prob_diff_by_candidate"] = {
            name: stats([r["model"]["vs_continuous"][name]["fall_prob_diff"] for r in records])
            for name in CONTINUOUS
        }
        agg["model_softmax_l1_diff_by_candidate"] = {
            name: stats([r["model"]["vs_continuous"][name]["softmax_l1_diff"] for r in records])
            for name in CONTINUOUS
        }
    return agg


def aggregate_by_subtype(records, have_model):
    """summary_by_subtype.csv 용 flat 집계."""
    rows = []
    by_st = {}
    for r in records:
        by_st.setdefault(r["subtype"], []).append(r)
    for st, rs in sorted(by_st.items()):
        amp = [rs_i["boundary"][b]["amp_jump_ratio"] for rs_i in rs for b in BOUNDARIES]
        spe = [rs_i["boundary"][b]["sparse_be_ratio"] for rs_i in rs for b in BOUNDARIES]
        srr = [rs_i["boundary"][b]["sdp_row_ratio"] for rs_i in rs for b in BOUNDARIES]
        zcos = [rs_i["pairwise"][n]["z_sdp_cosine"] for rs_i in rs for n in CONTINUOUS]
        row = {
            "subtype": st, "n_sessions": len(rs),
            "amp_jump_ratio_median": stats(amp).get("median"),
            "amp_jump_ratio_max": stats(amp).get("max"),
            "sparse_be_ratio_median": stats(spe).get("median"),
            "sparse_be_ratio_max": stats(spe).get("max"),
            "sdp_row_ratio_median": stats(srr).get("median"),
            "sdp_row_ratio_max": stats(srr).get("max"),
            "z_sdp_cosine_median": stats(zcos).get("median"),
            "z_sdp_cosine_min": stats(zcos).get("min"),
        }
        if have_model:
            fpd = [abs(rs_i["model"]["vs_continuous"][n]["fall_prob_diff"]) for rs_i in rs for n in CONTINUOUS]
            row["abs_fall_prob_diff_median"] = stats(fpd).get("median")
            row["abs_fall_prob_diff_max"] = stats(fpd).get("max")
        rows.append(row)
    return rows


def interpretation_flags(records, have_model):
    """판정 보조 플래그 (임계값 미적용)."""
    # 세션별 최대 boundary ratio (세 ratio × 두 boundary 중 max)
    ranked = []
    for r in records:
        mx = 0.0
        for b in BOUNDARIES:
            for k in ("amp_jump_ratio", "sparse_be_ratio", "sdp_row_ratio"):
                v = r["boundary"][b][k]
                if v is not None and np.isfinite(v):
                    mx = max(mx, float(v))
        ranked.append({"session": r["file"], "subtype": r["subtype"], "max_boundary_ratio": mx})
    ranked.sort(key=lambda x: x["max_boundary_ratio"], reverse=True)

    # 세션별 최저 z-SDP cosine (continuous 후보 중 min)
    lowsim = []
    for r in records:
        sims = [r["pairwise"][n]["z_sdp_cosine"] for n in CONTINUOUS]
        sims = [s for s in sims if s is not None and np.isfinite(s)]
        lowsim.append({"session": r["file"], "subtype": r["subtype"],
                       "min_z_sdp_cosine": float(min(sims)) if sims else None})
    lowsim.sort(key=lambda x: (x["min_z_sdp_cosine"] is None, x["min_z_sdp_cosine"]))

    return {
        "boundary_rows_available": True,  # 300f crop → boundary rows {3,4,5,23,24,25} 모두 <28
        "model_prob_available": bool(have_model),
        "high_boundary_ratio_ranked_sessions": ranked,
        "lowest_sdp_similarity_sessions": lowsim,
    }


# ─── CSV writers ─────────────────────────────────────────────────────────────
def write_selected_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:  # BOM: Excel(KR) Korean 정상표시
        w = csv.writer(fp)
        w.writerow(["subtype", "filename", "resampled_count", "decision", "is_substitute", "reason"])
        for r in records:
            w.writerow([r["subtype"], r["filename"], r["resampled_count"],
                        r["decision"], r["is_substitute"], r["reason"]])


def write_per_crop_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["subtype", "file", "crop", "crop_range",
                    "sdp_raw_peak_row", "sdp_z_peak_row",
                    "sdp_raw_energy_mean", "sdp_z_energy_mean",
                    "sparse_energy_peak_frame", "sparse_energy_mean"])
        for r in records:
            for name in ALL_CROPS:
                c = r["per_crop"][name]
                w.writerow([r["subtype"], r["file"], name,
                            "|".join(map(str, c["crop_range"])),
                            c["sdp_raw_peak_row"], c["sdp_z_peak_row"],
                            round(c["sdp_raw_energy_mean"], 6), round(c["sdp_z_energy_mean"], 6),
                            c["sparse_energy_peak_frame"], round(c["sparse_energy_mean"], 6)])


def write_pairwise_csv(path, records, have_model):
    head = ["subtype", "file", "candidate",
            "raw_sdp_cosine", "raw_sdp_mad", "z_sdp_cosine", "z_sdp_mad",
            "sdp_energy_curve_corr", "sdp_peak_row_diff", "sparse_peak_frame_diff"]
    if have_model:
        head += ["fall_prob_diff", "softmax_l1_diff"]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(head)
        for r in records:
            for name in CONTINUOUS:
                p = r["pairwise"][name]
                row = [r["subtype"], r["file"], name,
                       round(p["raw_sdp_cosine"], 5), round(p["raw_sdp_mad"], 6),
                       round(p["z_sdp_cosine"], 5), round(p["z_sdp_mad"], 6),
                       round(p["sdp_energy_curve_corr"], 5) if np.isfinite(p["sdp_energy_curve_corr"]) else "",
                       p["sdp_peak_row_diff"], p["sparse_peak_frame_diff"]]
                if have_model:
                    mc = r["model"]["vs_continuous"][name]
                    row += [round(mc["fall_prob_diff"], 5), round(mc["softmax_l1_diff"], 5)]
                w.writerow(row)


def write_boundary_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["subtype", "file", "boundary_b",
                    "amp_jump", "amp_nonbnd_median", "amp_jump_ratio",
                    "sparse_be_prev", "sparse_be", "sparse_be_next",
                    "sparse_nonbnd_median", "sparse_be_ratio",
                    "sdp_boundary_rows", "sdp_boundary_rows_energy_median",
                    "sdp_nonbnd_rows_median", "sdp_row_ratio"])
        for r in records:
            for b in BOUNDARIES:
                d = r["boundary"][b]
                w.writerow([r["subtype"], r["file"], b,
                            round(d["amp_jump"], 6), round(d["amp_nonbnd_median"], 6),
                            round(d["amp_jump_ratio"], 4) if d["amp_jump_ratio"] is not None else "",
                            round(d["sparse_be_prev"], 6), round(d["sparse_be"], 6), round(d["sparse_be_next"], 6),
                            round(d["sparse_nonbnd_median"], 6),
                            round(d["sparse_be_ratio"], 4) if d["sparse_be_ratio"] is not None else "",
                            "|".join(map(str, d["sdp_boundary_rows"])),
                            round(d["sdp_boundary_rows_energy_median"], 6),
                            round(d["sdp_nonbnd_rows_median"], 6),
                            round(d["sdp_row_ratio"], 4) if d["sdp_row_ratio"] is not None else ""])


def write_subtype_csv(path, rows, have_model):
    base = ["subtype", "n_sessions",
            "amp_jump_ratio_median", "amp_jump_ratio_max",
            "sparse_be_ratio_median", "sparse_be_ratio_max",
            "sdp_row_ratio_median", "sdp_row_ratio_max",
            "z_sdp_cosine_median", "z_sdp_cosine_min"]
    if have_model:
        base += ["abs_fall_prob_diff_median", "abs_fall_prob_diff_max"]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(base)
        for row in rows:
            w.writerow([row.get(k) for k in base])


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="게이트 1: beep concat artifact sanity 진단 (read-only)")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="6-class pretrained6 계열 checkpoint (.pt). 주면 모델 확률 비교 포함.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-plots", action="store_true", help="PNG 산출 생략")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT / "plots"
    arrays_dir = OUT / "arrays"

    # 모델 (선택)
    model, model_info, device = None, None, None
    if args.ckpt is not None:
        import torch
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if not args.ckpt.exists():
            raise FileNotFoundError(f"--ckpt 경로 없음: {args.ckpt}")
        model, model_info = load_pretrained6_model(args.ckpt, device)
        print(f"[ckpt] {args.ckpt}  out_dim={model_info['output_dim']} "
              f"classes={model_info['classes']} device={device}")
    have_model = model is not None

    if not HAVE_MPL:
        print(f"[WARN] matplotlib 미사용 ({_MPL_ERR}) → PNG 생략, 수치만 산출")

    # ── 세션 선택 ────────────────────────────────────────────────────────────
    print("\n=== 세션 선택 (resampled_count >= 550) ===")
    selected, sel_records = select_sessions()
    write_selected_csv(OUT / "selected_sessions.csv", sel_records)
    if not selected:
        print("[ERROR] 선택된 세션 없음. 종료.")
        return

    # ── 분석 ──────────────────────────────────────────────────────────────────
    print("\n=== 분석 ===")
    records = []
    do_plots = HAVE_MPL and not args.no_plots
    if do_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    for sess in selected:
        rec = analyze_session(sess, model, device)
        # 접합부 요약 한 줄
        b0 = rec["boundary"][BOUNDARIES[0]]
        b1 = rec["boundary"][BOUNDARIES[1]]
        msg = (f"[an] {rec['subtype']:12s} {rec['file']:32s} n={rec['n_frames']} | "
               f"amp_ratio {b0['amp_jump_ratio']:.2f}/{b1['amp_jump_ratio']:.2f} | "
               f"sdp_row_ratio {b0['sdp_row_ratio']:.2f}/{b1['sdp_row_ratio']:.2f}")
        if have_model:
            fp = rec["model"]["fall_prob"]
            msg += f" | fall cm={fp['concat_main']:.3f}"
        print(msg)

        # npz (raw/z SDP, sparse frame energy, probs)
        npz_payload = {f"sdp_z__{n}": rec["_arrays"]["sdp_z"][n] for n in ALL_CROPS}
        npz_payload.update({f"sdp_raw__{n}": rec["_arrays"]["sdp_raw"][n] for n in ALL_CROPS})
        npz_payload["sparse_frame_energy_concat"] = rec["_arrays"]["sparse_frame_energy_concat"]
        npz_payload["sdp_row_energy_concat_z"] = rec["_arrays"]["sdp_row_energy_concat_z"]
        if have_model:
            npz_payload.update({f"prob__{n}": rec["_arrays"]["probs"][n] for n in ALL_CROPS})
        np.savez_compressed(arrays_dir / f"{rec['subtype']}__{Path(rec['file']).stem}.npz", **npz_payload)

        if do_plots:
            plot_session(rec, plots_dir, have_model)

        rec.pop("_arrays", None)  # summary.json 비대화 방지
        records.append(rec)

    # ── 산출물 ────────────────────────────────────────────────────────────────
    write_per_crop_csv(OUT / "per_crop_metrics.csv", records)
    write_pairwise_csv(OUT / "pairwise_comparison.csv", records, have_model)
    write_boundary_csv(OUT / "boundary_artifact_metrics.csv", records)

    agg = aggregate(records, have_model)
    st_rows = aggregate_by_subtype(records, have_model)
    write_subtype_csv(OUT / "summary_by_subtype.csv", st_rows, have_model)
    flags = interpretation_flags(records, have_model)

    summary = {
        "meta": {
            "purpose": "Gate 1 — beep 제거 concat 접합부 artifact sanity (read-only)",
            "note": ("continuous 후보는 concat_main 과 동일 내용 1:1 대응이 아니라 "
                     "beep 미제거 '대체 crop 후보' 비교임."),
            "preprocess": {
                "rpca_max_iter": DEFAULT_MAX_ITER, "rpca_tol": None,
                "sdp_sub_w": SUB_W, "sdp_stride": SUB_STRIDE, "n_lags": N_LAGS,
                "row_energy_def": "mean(abs(SDP[row,:]))",
                "frame_energy_def": "mean(abs(mat[t,:]))",
            },
            "coords": {
                "clean400": "original[50:150]+[200:400]+[450:550]",
                "concat_main": "clean[50:350] = original[100:150]+[200:400]+[450:500]",
                "boundaries_crop_local": BOUNDARIES,
                "continuous_candidates": {k: list(v) for k, v in CONTINUOUS.items()},
            },
            "min_resampled_count": MIN_FRAMES,
            "per_subtype": PER_SUBTYPE,
            "subtypes": SUBTYPES,
            "n_sessions": len(records),
            "model": model_info,
        },
        "aggregate_overall": agg,
        "aggregate_by_subtype": st_rows,
        "interpretation_flags": flags,
        "sessions": records,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 산출 완료 ===")
    for f in ["selected_sessions.csv", "per_crop_metrics.csv", "pairwise_comparison.csv",
              "boundary_artifact_metrics.csv", "summary_by_subtype.csv", "summary.json"]:
        print(f"  {OUT / f}")
    if do_plots:
        print(f"  {plots_dir}\\*.png  (세션 {len(records)}개 × 3~4장)")
    print(f"  {arrays_dir}\\*.npz")
    print("\n[flags] high_boundary_ratio_ranked top3:",
          [(x['session'], round(x['max_boundary_ratio'], 2)) for x in flags["high_boundary_ratio_ranked_sessions"][:3]])
    print("[flags] lowest_sdp_similarity top3:",
          [(x['session'], None if x['min_z_sdp_cosine'] is None else round(x['min_z_sdp_cosine'], 3))
           for x in flags["lowest_sdp_similarity_sessions"][:3]])


if __name__ == "__main__":
    main()
