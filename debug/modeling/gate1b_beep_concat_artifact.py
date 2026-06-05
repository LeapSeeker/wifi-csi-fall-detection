"""게이트 1-b: beep concat artifact 보강 진단 (read-only).

게이트 1 1차 결론: continuous_center = original[150:450] 가 1순위 fallback 후보.
다만 concat 기각 근거가 SDP/모델 레벨에서 부족(amplitude boundary는 튀지만 RPCA
sparse/SDP dominant peak는 실제 transient에 위치). D-031 좌표계 수정 전, concat 이
모델 레벨에서 실제 artifact 를 만드는지 보강 진단한다.

3축 진단:
  1. 모델 확률 비교 : concat_main 에서만 (WALK) fall prob / thr0.1 crossing 이 튀는지.
  2. attention      : 모델이 concat boundary rows({3,4,5,23,24,25})를 실제로 보는지.
  3. 전 subject/env : WALK boundary artifact 가 E1_S01 특성인지, 전반에 재현되는지.

좌표계 (게이트 1 과 동일)
  resampled original = amp[:550]
  clean = original[50:150] + original[200:400] + original[450:550]   # 400f
  concat_main = clean[50:350]                                         # 300f
              = original[100:150] + original[200:400] + original[450:500]
    접합부 crop-local 50(=orig149->200), 250(=orig399->450)
  continuous_pre_fall  = original[100:400]
  continuous_center    = original[150:450]
  continuous_fall_post = original[200:500]

전처리 경로 (정식 window_to_model_input 과 동일, 1회 consistency 검증)
  crop(300,n_sc) -> rpca_sparse(200,None) -> stacked_doppler_profile(30,10,20)
  -> (28,20) -> global z-score -> (1,28,20) -> (1,1,28,20) -> model -> softmax

판정 임계값은 박지 않는다. 수치/요약만 산출하고 최종 판정은 진규/Codex/Claude 검토.
단 thr0.1 crossing 은 baseline-axis 운영점과 직결되므로 보조 플래그로만 저장.

제약: 동결 파일(pipeline/rpca/acf/sdp/학습·추론) 및 데이터 read-only.
      산출물은 debug/modeling/diag_out/beep_concat_artifact/gate1b/ 하위에만 기록.

사용법:
  python debug/modeling/gate1b_beep_concat_artifact.py            # 기본 ckpt 자동
  python debug/modeling/gate1b_beep_concat_artifact.py --ckpt A.pt --ckpt B.pt
  python debug/modeling/gate1b_beep_concat_artifact.py --no-model # SDP-only
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

from model.preprocessing.loader import (                            # noqa: E402
    load_safesignal_csv, parse_safesignal_filename,
)
from model.preprocessing.resample import resample_to_100hz          # noqa: E402
from model.preprocessing.rpca import rpca_sparse, DEFAULT_MAX_ITER  # noqa: E402
from model.preprocessing.sdp import (                               # noqa: E402
    stacked_doppler_profile, SUB_W, SUB_STRIDE, W_T,
)
from model.preprocessing.acf import N_LAGS                          # noqa: E402
from model.preprocessing.pipeline import window_to_model_input      # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as _e:  # pragma: no cover
    HAVE_MPL = False
    _MPL_ERR = repr(_e)

OUT = ROOT / "debug/modeling/diag_out/beep_concat_artifact/gate1b"
CLEANED = ROOT / "data/cleaned"

PRIMARY_SUBTYPES = ["FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B", "FALL_WALK_F", "FALL_WALK_B"]
WALK_SUBTYPES = ["FALL_WALK_F", "FALL_WALK_B"]
PER_SUBTYPE_PRIMARY = 2
WALK_CROSS_MIN = 2
WALK_CROSS_QUOTA = 4
MIN_FRAMES = 550
THR = 0.1  # baseline-axis 운영점 보조 플래그 (좌표계 판정 임계값 아님)

BOUNDARIES = [50, 250]
CONTINUOUS = {
    "continuous_pre_fall": (100, 400),
    "continuous_center": (150, 450),
    "continuous_fall_post": (200, 500),
}
ALL_CROPS = ["concat_main"] + list(CONTINUOUS.keys())
COORD_POLICY = {
    "concat_main": "clean400_concat",
    "continuous_pre_fall": "continuous_pre_fall",
    "continuous_center": "continuous_center",
    "continuous_fall_post": "continuous_fall_post",
}

# 6-class pretrained6 라벨 (fall index 0)
PRETRAINED6_CLASSES = ("fall", "walking", "sit_stand", "lying", "standing", "picking")


# ─── SDP bin ↔ frame 매핑 ────────────────────────────────────────────────────
def sdp_rows_for_frame(f, sub_w=SUB_W, stride=SUB_STRIDE, n_rows=W_T):
    """frame f 를 포함하는 SDP row 집합. row i = frame [stride*i, stride*i+sub_w).
    b=50 -> {3,4,5}, b=250 -> {23,24,25}. 0<=row<n_rows 만."""
    lo = int(np.ceil((f - (sub_w - 1)) / stride))
    hi = int(np.floor(f / stride))
    return [i for i in range(max(0, lo), min(n_rows - 1, hi) + 1)]


BOUNDARY_ROWS = sorted(set(sum((sdp_rows_for_frame(b) for b in BOUNDARIES), [])))  # {3,4,5,23,24,25}


# ─── 전처리 경로 헬퍼 ────────────────────────────────────────────────────────
def zscore(sdp):
    return (sdp - sdp.mean()) / (sdp.std() + 1e-6)


def row_energy(sdp):
    return np.abs(sdp).mean(axis=1)  # (28,)


def frame_energy(mat):
    return np.abs(mat).mean(axis=1)  # (n_t,)


def preprocess_crop(crop):
    """crop(300,n_sc) → dict(sparse, sdp_raw, sdp_z, model_input(1,28,20)).
    정식 window_to_model_input 내부와 동일 경로."""
    sparse = rpca_sparse(crop, max_iter=DEFAULT_MAX_ITER, tol=None)
    sdp_raw = stacked_doppler_profile(sparse, sub_w=SUB_W, stride=SUB_STRIDE, n_lags=N_LAGS)
    sdp_z = zscore(sdp_raw)
    return {"sparse": sparse, "sdp_raw": sdp_raw, "sdp_z": sdp_z, "model_input": sdp_z[None, ...]}


def cosine(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else float("nan")


def mad(a, b):
    return float(np.mean(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def pearson(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan")


# ─── crop 구성 ───────────────────────────────────────────────────────────────
def build_crops(A550):
    clean = np.concatenate([A550[50:150], A550[200:400], A550[450:550]], axis=0)  # (400,dim)
    crops = {"concat_main": clean[50:350]}
    for name, (lo, hi) in CONTINUOUS.items():
        crops[name] = A550[lo:hi]
    return crops


# ─── 모델 ────────────────────────────────────────────────────────────────────
def default_checkpoints():
    """기본 checkpoint 후보 (존재하는 것만)."""
    out = []
    p1 = ROOT / "model/pretrained/checkpoints/best.pt"
    if p1.exists():
        out.append(("pretrained_zero_shot", p1))
    p2 = ROOT / "model/finetune/checkpoints_compare6_cpu/best_operating.pt"
    if p2.exists():
        out.append(("finetuned_baseline6", p2))
    return out


def load_pretrained6_model(ckpt_path, device):
    """6-class pretrained6 계열 checkpoint 로드 + 검증. (model, info, supports_attention)."""
    import torch
    from model.pretrained.model import CNNGRUAttention

    ck = torch.load(ckpt_path, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    out_w = None
    for key in ("classifier.1.weight", "classifier.weight"):
        if key in sd:
            out_w = sd[key]; break
    if out_w is None:
        cand = [k for k in sd if k.endswith("weight") and sd[k].ndim == 2]
        out_w = sd[cand[-1]] if cand else None
    out_dim = int(out_w.shape[0]) if out_w is not None else None
    if out_dim is not None and out_dim != 6:
        raise ValueError(f"checkpoint output dim != 6 (got {out_dim}).")

    classes = ck.get("classes") if isinstance(ck, dict) else None
    if classes is not None:
        classes = list(classes)
        if len(classes) != 6 or classes[0] != "fall":
            raise ValueError(f"checkpoint classes 검증 실패 (got {classes}). classes[0]=='fall' & len==6 필요.")

    model = CNNGRUAttention(n_classes=6).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    with torch.no_grad():
        o = model(torch.zeros(1, 1, W_T, N_LAGS, device=device))
    if int(o.shape[1]) != 6:
        raise ValueError(f"forward output dim != 6 (got {int(o.shape[1])}).")

    # attention 지원 여부 탐지
    supports_attention = False
    try:
        with torch.no_grad():
            res = model(torch.zeros(1, 1, W_T, N_LAGS, device=device), return_attention=True)
        supports_attention = isinstance(res, (tuple, list)) and len(res) == 2
    except Exception:
        supports_attention = False

    info = {
        "ckpt_path": str(ckpt_path),
        "output_dim": out_dim if out_dim is not None else 6,
        "classes": classes,
        "class_policy": ck.get("class_policy") if isinstance(ck, dict) else None,
        "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
        "saved_threshold": ck.get("threshold") if isinstance(ck, dict) else None,
        "supports_attention": supports_attention,
    }
    return model, info, supports_attention


def infer(model, model_input, device, want_attention):
    """model_input (1,28,20) → (softmax probs (6,), attn (28,) or None)."""
    import torch
    xb = torch.as_tensor(model_input[None, ...], dtype=torch.float32).to(device)  # (1,1,28,20)
    with torch.no_grad():
        if want_attention:
            logits, weights = model(xb, return_attention=True)
            attn = weights.squeeze(0).cpu().numpy()  # (28,)
        else:
            logits = model(xb)
            attn = None
        p = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return p, attn


# ─── 세션 선택 ───────────────────────────────────────────────────────────────
def _meta_of(f):
    m = parse_safesignal_filename(f)
    return m.environment, m.subject, m.trial


def _resampled_count(f):
    raw = load_safesignal_csv(f, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    return int(res.resampled_count)


def select_primary():
    """E1_S01 6 subtype × 2 (resampled_count>=550). (sessions, records)."""
    sessions, records = [], []
    for st in PRIMARY_SUBTYPES:
        files = sorted(CLEANED.rglob(f"*{st}*_T*.csv"))
        files = [f for f in files if "E1_S01" in f.name]  # primary 는 E1_S01 한정
        got = 0
        for f in files:
            try:
                env, subj, trial = _meta_of(f)
                n = _resampled_count(f)
            except Exception as e:
                records.append(dict(file=f.name, subtype=st, env="", subject="", trial="",
                                    resampled_count="", selected=False,
                                    selection_group="candidate_skipped_parse_error", skip_reason=repr(e)[:120]))
                continue
            base = dict(file=f.name, subtype=st, env=env, subject=subj, trial=trial, resampled_count=n)
            if n < MIN_FRAMES:
                records.append({**base, "selected": False,
                                "selection_group": "candidate_skipped_short", "skip_reason": f"n<{MIN_FRAMES}"})
                continue
            if got >= PER_SUBTYPE_PRIMARY:
                records.append({**base, "selected": False,
                                "selection_group": "candidate_skipped_over_quota", "skip_reason": "primary quota=2"})
                continue
            sessions.append({"path": f, "subtype": st, "env": env, "subject": subj, "trial": trial,
                             "resampled_count": n, "selection_group": "primary_e1_s01"})
            records.append({**base, "selected": True, "selection_group": "primary_e1_s01", "skip_reason": ""})
            print(f"[primary] {st} {f.name} n={n}")
            got += 1
        if got < PER_SUBTYPE_PRIMARY:
            print(f"[WARN] primary {st}: {got}/{PER_SUBTYPE_PRIMARY}")
    return sessions, records


def select_walk_cross_subject(primary_paths):
    """전 subject/env WALK 추가. diversity(env,subject) 우선, >=550, subtype당 최소2 최대4.
    cross-subject 후보 스캔 전체를 기록. (sessions, records)."""
    sessions, records = [], []
    for st in WALK_SUBTYPES:
        files = sorted(CLEANED.rglob(f"*{st}*_T*.csv"))
        eligible = []  # (f, env, subj, trial, n)
        for f in files:
            if str(f) in primary_paths or ("E1_S01" in f.name):
                continue  # 기존 primary 제외
            try:
                env, subj, trial = _meta_of(f)
            except Exception as e:
                records.append(dict(file=f.name, subtype=st, env="", subject="", trial="",
                                    resampled_count="", selected=False,
                                    selection_group="candidate_skipped_parse_error", skip_reason=repr(e)[:120]))
                continue
            try:
                n = _resampled_count(f)
            except Exception as e:
                records.append(dict(file=f.name, subtype=st, env=env, subject=subj, trial=trial,
                                    resampled_count="", selected=False,
                                    selection_group="candidate_skipped_parse_error", skip_reason=repr(e)[:120]))
                continue
            if n < MIN_FRAMES:
                records.append(dict(file=f.name, subtype=st, env=env, subject=subj, trial=trial,
                                    resampled_count=n, selected=False,
                                    selection_group="candidate_skipped_short", skip_reason=f"n<{MIN_FRAMES}"))
                continue
            eligible.append((f, env, subj, trial, n))

        # pass A: diversity (distinct (env,subject)) up to quota
        chosen_paths, combos = set(), set()
        for (f, env, subj, trial, n) in eligible:
            if len(chosen_paths) >= WALK_CROSS_QUOTA:
                break
            if (env, subj) in combos:
                continue
            combos.add((env, subj)); chosen_paths.add(str(f))
        # pass B: fill to min with duplicates if diversity insufficient
        if len(chosen_paths) < WALK_CROSS_MIN:
            for (f, env, subj, trial, n) in eligible:
                if len(chosen_paths) >= WALK_CROSS_MIN:
                    break
                if str(f) in chosen_paths:
                    continue
                chosen_paths.add(str(f)); combos.add((env, subj))

        for (f, env, subj, trial, n) in eligible:
            base = dict(file=f.name, subtype=st, env=env, subject=subj, trial=trial, resampled_count=n)
            if str(f) in chosen_paths:
                sessions.append({"path": f, "subtype": st, "env": env, "subject": subj, "trial": trial,
                                 "resampled_count": n, "selection_group": "walk_cross_subject"})
                records.append({**base, "selected": True, "selection_group": "walk_cross_subject", "skip_reason": ""})
                print(f"[walk-x] {st} {f.name} n={n} (E{env}_S{subj:02d})")
            elif (env, subj) in combos:
                records.append({**base, "selected": False,
                                "selection_group": "candidate_skipped_duplicate_subject_env",
                                "skip_reason": f"(env,subj)=({env},{subj}) 이미 선택"})
            else:
                records.append({**base, "selected": False,
                                "selection_group": "candidate_skipped_over_quota",
                                "skip_reason": f"quota={WALK_CROSS_QUOTA} 초과"})
        n_sel = sum(1 for r in records if r["subtype"] == st and r["selected"])
        if n_sel < WALK_CROSS_MIN:
            print(f"[WARN] walk-cross {st}: {n_sel}/{WALK_CROSS_MIN} (>=550 후보 부족)")
    return sessions, records


# ─── 세션 분석 ───────────────────────────────────────────────────────────────
def analyze_session(sess, checkpoints, device, consistency_box):
    raw = load_safesignal_csv(sess["path"], rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    A = res.amplitude
    n = int(res.resampled_count)
    dim = int(A.shape[1])
    A550 = A[:550]
    crops = build_crops(A550)
    pp = {name: preprocess_crop(crop) for name, crop in crops.items()}

    # 1회 consistency: preprocess_crop.model_input == window_to_model_input
    if not consistency_box["done"]:
        mi_ref = window_to_model_input(crops["concat_main"])
        consistency_box["ok"] = bool(np.allclose(pp["concat_main"]["model_input"][0], mi_ref, atol=1e-4))
        consistency_box["done"] = True

    # ── per-crop SDP 스칼라 + pairwise(vs concat) ────────────────────────────
    cm = pp["concat_main"]
    cm_re_raw = row_energy(cm["sdp_raw"]); cm_fe = frame_energy(cm["sparse"])
    per_crop, pairwise = {}, {}
    for name in ALL_CROPS:
        d = pp[name]
        re_raw = row_energy(d["sdp_raw"]); fe = frame_energy(d["sparse"])
        per_crop[name] = {
            "sdp_raw_peak_row": int(re_raw.argmax()),
            "sdp_z_peak_row": int(row_energy(d["sdp_z"]).argmax()),
            "sparse_energy_peak_frame": int(fe.argmax()),
        }
        if name == "concat_main":
            pairwise[name] = {"raw_sdp_cosine": 1.0, "raw_sdp_mad": 0.0, "z_sdp_cosine": 1.0,
                              "z_sdp_mad": 0.0, "energy_curve_corr": 1.0,
                              "sdp_peak_row_diff": 0, "sparse_peak_frame_diff": 0}
        else:
            pairwise[name] = {
                "raw_sdp_cosine": cosine(cm["sdp_raw"], d["sdp_raw"]),
                "raw_sdp_mad": mad(cm["sdp_raw"], d["sdp_raw"]),
                "z_sdp_cosine": cosine(cm["sdp_z"], d["sdp_z"]),
                "z_sdp_mad": mad(cm["sdp_z"], d["sdp_z"]),
                "energy_curve_corr": pearson(cm_re_raw, re_raw),
                "sdp_peak_row_diff": int(abs(int(cm_re_raw.argmax()) - int(re_raw.argmax()))),
                "sparse_peak_frame_diff": int(abs(int(cm_fe.argmax()) - int(fe.argmax()))),
            }

    # ── 접합부 artifact (concat_main) ────────────────────────────────────────
    crop = crops["concat_main"]
    adj = np.abs(np.diff(crop, axis=0)).mean(axis=1)  # (299,)
    nonbnd_adj = np.delete(adj, [b - 1 for b in BOUNDARIES])
    adj_med = float(np.median(nonbnd_adj))
    nbr = set(sum(([b - 1, b, b + 1] for b in BOUNDARIES), []))
    nonbnd_fe = np.array([cm_fe[t] for t in range(len(cm_fe)) if t not in nbr])
    fe_med = float(np.median(nonbnd_fe))
    re_z_cm = row_energy(cm["sdp_z"])
    nonbnd_rows = [i for i in range(W_T) if i not in BOUNDARY_ROWS]
    rows_med = float(np.median(re_z_cm[nonbnd_rows]))
    boundary = {}
    for b in BOUNDARIES:
        rows_b = sdp_rows_for_frame(b)
        rb_med = float(np.median(re_z_cm[rows_b]))
        boundary[b] = {
            "amp_jump": float(adj[b - 1]), "amp_nonbnd_median": adj_med,
            "amp_jump_ratio": float(adj[b - 1] / adj_med) if adj_med else None,
            "sparse_be": float(cm_fe[b]), "sparse_nonbnd_median": fe_med,
            "sparse_be_ratio": float(cm_fe[b] / fe_med) if fe_med else None,
            "sdp_boundary_rows": rows_b, "sdp_boundary_rows_energy_median": rb_med,
            "sdp_nonbnd_rows_median": rows_med,
            "sdp_row_ratio": float(rb_med / rows_med) if rows_med else None,
        }

    # ── 모델 확률 + attention (ckpt별) ───────────────────────────────────────
    model_by_ckpt = {}
    arrays_probs, arrays_attn = {}, {}
    for ck_id, model, supports_attn in checkpoints:
        probs, attns = {}, {}
        for name in ALL_CROPS:
            p, attn = infer(model, pp[name]["model_input"], device, supports_attn)
            probs[name] = p
            attns[name] = attn
        cm_p = probs["concat_main"]
        cm_cross = bool(cm_p[0] >= THR)
        crop_fields = {}
        for name in ALL_CROPS:
            p = probs[name]
            fall = float(p[0])
            max_nonfall = float(p[1:].max())
            attn = attns[name]
            if attn is not None:
                a = np.asarray(attn, np.float64)
                b_mass = float(a[BOUNDARY_ROWS].sum())
                tot = float(a.sum())
                att = {
                    "attention_available": True,
                    "attention_boundary_mass": b_mass,
                    "attention_total_mass": tot,
                    "attention_boundary_ratio": float(b_mass / tot) if tot else None,
                    "attention_peak_row": int(a.argmax()),
                    "attention_peak_value": float(a.max()),
                }
            else:
                att = {"attention_available": False, "attention_boundary_mass": None,
                       "attention_total_mass": None, "attention_boundary_ratio": None,
                       "attention_peak_row": None, "attention_peak_value": None}
            crop_fields[name] = {
                "fall_prob": fall,
                "pred_class": PRETRAINED6_CLASSES[int(p.argmax())],
                "fall_rank": int((p > p[0]).sum()) + 1,
                "max_nonfall_prob": max_nonfall,
                "fall_margin": float(fall - max_nonfall),
                "crosses_thr_0p1": bool(fall >= THR),
                "softmax_l1_diff_vs_concat": (0.0 if name == "concat_main"
                                              else float(np.abs(cm_p - p).sum())),
                "fall_prob_diff_concat_minus_candidate": (None if name == "concat_main"
                                                          else float(cm_p[0] - fall)),
                "concat_crosses_but_candidate_not": (None if name == "concat_main"
                                                     else bool(cm_cross and not (fall >= THR))),
                **att,
            }
        model_by_ckpt[ck_id] = crop_fields
        arrays_probs[ck_id] = probs
        arrays_attn[ck_id] = attns

    return {
        "file": sess["path"].name, "subtype": sess["subtype"],
        "env": sess["env"], "subject": sess["subject"], "trial": sess["trial"],
        "selection_group": sess["selection_group"], "n_frames": n, "dim": dim,
        "per_crop": per_crop, "pairwise": pairwise, "boundary": boundary,
        "model_by_ckpt": model_by_ckpt,
        "_arrays": {"sdp_z": {n: pp[n]["sdp_z"] for n in ALL_CROPS},
                    "sparse_frame_energy_concat": cm_fe,
                    "sdp_row_energy_concat_z": re_z_cm,
                    "probs": arrays_probs, "attn": arrays_attn},
    }


# ─── metric row 평탄화 (per_candidate_metrics.csv / 집계 공용) ────────────────
def build_metric_rows(records, ckpt_ids, have_model):
    rows = []
    for r in records:
        for ck_id in (ckpt_ids if have_model else ["(none)"]):
            for cand in ALL_CROPS:
                pw = r["pairwise"][cand]
                row = {
                    "file": r["file"], "subtype": r["subtype"], "env": r["env"],
                    "subject": r["subject"], "trial": r["trial"],
                    "selection_group": r["selection_group"],
                    "checkpoint_id": ck_id, "model_available": have_model,
                    "candidate": cand, "coordinate_policy": COORD_POLICY[cand],
                    "raw_sdp_cosine": pw["raw_sdp_cosine"], "raw_sdp_mad": pw["raw_sdp_mad"],
                    "z_sdp_cosine": pw["z_sdp_cosine"], "z_sdp_mad": pw["z_sdp_mad"],
                    "energy_curve_corr": pw["energy_curve_corr"],
                    "sdp_peak_row_diff": pw["sdp_peak_row_diff"],
                    "sparse_peak_frame_diff": pw["sparse_peak_frame_diff"],
                }
                if have_model:
                    m = r["model_by_ckpt"][ck_id][cand]
                    row.update(m)
                rows.append(row)
    return rows


# ─── 집계 ────────────────────────────────────────────────────────────────────
def stats(vals, want=("median", "min", "max")):
    a = np.asarray([v for v in vals if v is not None and (not isinstance(v, bool))], np.float64)
    a = a[np.isfinite(a)]
    out = {}
    if a.size == 0:
        return {k: None for k in want}
    fn = {"median": np.median, "min": np.min, "max": np.max,
          "p25": lambda x: np.percentile(x, 25), "p75": lambda x: np.percentile(x, 75),
          "mean": np.mean}
    for k in want:
        out[k] = float(fn[k](a))
    return out


def aggregate_by_subtype_by_candidate(metric_rows, have_model):
    groups = {}
    for row in metric_rows:
        key = (row["subtype"], row["candidate"], row["checkpoint_id"])
        groups.setdefault(key, []).append(row)
    out = []
    for (subtype, candidate, ck_id), rs in sorted(groups.items()):
        rec = {
            "subtype": subtype, "candidate": candidate, "checkpoint_id": ck_id,
            "model_available": have_model, "n_sessions": len(rs),
            "z_sdp_cosine_median": stats([r["z_sdp_cosine"] for r in rs])["median"],
            "z_sdp_cosine_min": stats([r["z_sdp_cosine"] for r in rs])["min"],
            "raw_sdp_cosine_median": stats([r["raw_sdp_cosine"] for r in rs])["median"],
            "raw_sdp_cosine_min": stats([r["raw_sdp_cosine"] for r in rs])["min"],
            "energy_curve_corr_median": stats([r["energy_curve_corr"] for r in rs])["median"],
            "energy_curve_corr_min": stats([r["energy_curve_corr"] for r in rs])["min"],
            "sdp_peak_row_diff_median": stats([r["sdp_peak_row_diff"] for r in rs])["median"],
            "sdp_peak_row_diff_max": stats([r["sdp_peak_row_diff"] for r in rs])["max"],
            "sparse_peak_frame_diff_median": stats([r["sparse_peak_frame_diff"] for r in rs])["median"],
            "sparse_peak_frame_diff_max": stats([r["sparse_peak_frame_diff"] for r in rs])["max"],
        }
        if have_model:
            rec.update({
                "fall_prob_median": stats([r["fall_prob"] for r in rs])["median"],
                "fall_prob_max": stats([r["fall_prob"] for r in rs])["max"],
                "fall_prob_diff_concat_minus_candidate_median":
                    stats([r["fall_prob_diff_concat_minus_candidate"] for r in rs])["median"],
                "fall_prob_diff_concat_minus_candidate_max":
                    stats([r["fall_prob_diff_concat_minus_candidate"] for r in rs])["max"],
                "crosses_thr_0p1_count": int(sum(1 for r in rs if r.get("crosses_thr_0p1"))),
                "concat_crosses_but_candidate_not_count":
                    int(sum(1 for r in rs if r.get("concat_crosses_but_candidate_not"))),
                "attention_boundary_mass_median": stats([r["attention_boundary_mass"] for r in rs])["median"],
                "attention_boundary_mass_max": stats([r["attention_boundary_mass"] for r in rs])["max"],
                "attention_boundary_ratio_median": stats([r["attention_boundary_ratio"] for r in rs])["median"],
                "attention_boundary_ratio_max": stats([r["attention_boundary_ratio"] for r in rs])["max"],
            })
        out.append(rec)
    return out


# ─── CSV writers ─────────────────────────────────────────────────────────────
def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])


def round_rows(rows, ndigits=5):
    out = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            if isinstance(v, float) and np.isfinite(v):
                rr[k] = round(v, ndigits)
            elif v is None:
                rr[k] = ""
            else:
                rr[k] = v
        out.append(rr)
    return out


# ─── 그림 ────────────────────────────────────────────────────────────────────
def plot_session(rec, plots_dir, ckpt_ids, have_model):
    if not HAVE_MPL:
        return
    arr = rec["_arrays"]
    stem = f"{rec['subtype']}__E{rec['env']}_S{rec['subject']:02d}_T{rec['trial']:03d}"

    fe = arr["sparse_frame_energy_concat"]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(len(fe)), fe, lw=0.9)
    for b in BOUNDARIES:
        ax.axvline(b, color="C3", ls="--", lw=1.0)
    ax.set_title(f"{stem} | concat_main sparse frame energy")
    ax.set_xlabel("crop-local frame"); ax.set_ylabel("mean|sparse|")
    fig.tight_layout(); fig.savefig(plots_dir / f"{stem}__sparse_energy.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, len(ALL_CROPS), figsize=(3.2 * len(ALL_CROPS), 3.4))
    vmax = max(np.abs(arr["sdp_z"][n]).max() for n in ALL_CROPS)
    for ax, name in zip(axes, ALL_CROPS):
        im = ax.imshow(arr["sdp_z"][name], aspect="auto", origin="lower", cmap="magma", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=9); ax.set_xlabel("lag"); ax.set_ylabel("row")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025)
    fig.suptitle(f"{stem} | SDP(z) heatmaps", fontsize=10)
    fig.savefig(plots_dir / f"{stem}__sdp_heatmaps.png", dpi=110); plt.close(fig)

    if have_model:
        ck0 = ckpt_ids[0]
        # fall prob bar
        fp = {n: rec["model_by_ckpt"][ck0][n]["fall_prob"] for n in ALL_CROPS}
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(len(ALL_CROPS)), [fp[n] for n in ALL_CROPS])
        ax.axhline(THR, color="C3", ls="--", lw=1.0, label=f"thr {THR}")
        ax.set_xticks(range(len(ALL_CROPS))); ax.set_xticklabels(ALL_CROPS, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_ylabel("fall_prob"); ax.legend(fontsize=8)
        ax.set_title(f"{stem} | fall_prob [{ck0}]")
        fig.tight_layout(); fig.savefig(plots_dir / f"{stem}__fall_prob.png", dpi=110); plt.close(fig)
        # attention curves (if available)
        attn0 = rec["model_by_ckpt"][ck0]["concat_main"]
        if attn0["attention_available"]:
            fig, ax = plt.subplots(figsize=(9, 3))
            for name in ALL_CROPS:
                a = arr["attn"][ck0][name]
                ax.plot(np.arange(len(a)), a, marker="o", ms=2.5, lw=0.9, label=name)
            for i in BOUNDARY_ROWS:
                ax.axvline(i, color="0.6", ls=":", lw=0.7)
            ax.set_title(f"{stem} | attention over rows  (boundary rows {BOUNDARY_ROWS}) [{ck0}]")
            ax.set_xlabel("SDP row"); ax.set_ylabel("attention weight"); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(plots_dir / f"{stem}__attention.png", dpi=110); plt.close(fig)


# ─── share summary (수치 텍스트, 섹션 6) ─────────────────────────────────────
def build_share_summary(records, agg, ckpt_ids, have_model, attn_available):
    L = []
    L.append("# 게이트 1-b 수치 요약 (공유용)\n")
    L.append(f"- 분석 세션: {len(records)} (primary E1_S01 + walk cross-subject)")
    L.append(f"- checkpoints: {ckpt_ids if have_model else '없음 (SDP-only)'}")
    L.append(f"- attention 사용 가능: {attn_available}\n")

    def agg_get(subtype, candidate, ck_id, key):
        for a in agg:
            if a["subtype"] == subtype and a["candidate"] == candidate and a["checkpoint_id"] == ck_id:
                return a.get(key)
        return None

    if have_model:
        L.append("## 1) concat_main vs continuous fall_prob (subtype별, ckpt별)")
        for ck_id in ckpt_ids:
            L.append(f"\n### ckpt = {ck_id}")
            L.append("subtype | crop | fall_prob_median(max) | crosses_thr0.1_cnt | concat_only_cross_cnt")
            subtypes = sorted({r["subtype"] for r in records})
            for st in subtypes:
                for cand in ALL_CROPS:
                    fm = agg_get(st, cand, ck_id, "fall_prob_median")
                    fx = agg_get(st, cand, ck_id, "fall_prob_max")
                    cc = agg_get(st, cand, ck_id, "crosses_thr_0p1_count")
                    co = agg_get(st, cand, ck_id, "concat_crosses_but_candidate_not_count")
                    fm_s = "n/a" if fm is None else f"{fm:.3f}"
                    fx_s = "n/a" if fx is None else f"{fx:.3f}"
                    L.append(f"{st} | {cand} | {fm_s}({fx_s}) | {cc} | {co if cand!='concat_main' else '-'}")
        L.append("\n## 2) WALK: concat_main fall_prob/thr-crossing 이 continuous 대비 튀는가")
        for ck_id in ckpt_ids:
            for st in ["FALL_WALK_F", "FALL_WALK_B"]:
                cm = agg_get(st, "concat_main", ck_id, "fall_prob_median")
                cc = agg_get(st, "continuous_center", ck_id, "fall_prob_median")
                cm_cross = agg_get(st, "concat_main", ck_id, "crosses_thr_0p1_count")
                cc_cross = agg_get(st, "continuous_center", ck_id, "crosses_thr_0p1_count")
                if cm is None:
                    continue
                spike = (cm > cc) if (cm is not None and cc is not None) else None
                L.append(f"- [{ck_id}] {st}: concat fall_med={cm:.3f} vs center={cc:.3f} "
                         f"→ concat 우세={spike} | thr-crossing concat={cm_cross} center={cc_cross}")

    if have_model and attn_available:
        L.append("\n## 3) attention: concat boundary mass 가 continuous 대비 높은가")
        for ck_id in ckpt_ids:
            for st in ["FALL_WALK_F", "FALL_WALK_B"]:
                cm = agg_get(st, "concat_main", ck_id, "attention_boundary_ratio_median")
                cc = agg_get(st, "continuous_center", ck_id, "attention_boundary_ratio_median")
                if cm is None:
                    continue
                hi = (cm > cc) if (cm is not None and cc is not None) else None
                L.append(f"- [{ck_id}] {st}: concat boundary_ratio_med={cm:.3f} vs center={cc:.3f} → concat 높음={hi}")
    elif have_model:
        L.append("\n## 3) attention: 모델이 attention 반환 미지원 → skip")

    L.append("\n## 4) cross-subject WALK boundary artifact vs E1_S01 WALK")
    L.append("group | subtype | env_subj | amp_jump_ratio(b50/b250) | sparse_be_ratio | sdp_row_ratio")
    for r in records:
        if not r["subtype"].startswith("FALL_WALK"):
            continue
        b0, b1 = r["boundary"][50], r["boundary"][250]
        grp = "E1_S01" if r["selection_group"] == "primary_e1_s01" else f"E{r['env']}_S{r['subject']:02d}"
        L.append(f"{r['selection_group']} | {r['subtype']} | {grp} | "
                 f"{b0['amp_jump_ratio']:.2f}/{b1['amp_jump_ratio']:.2f} | "
                 f"{b0['sparse_be_ratio']:.2f}/{b1['sparse_be_ratio']:.2f} | "
                 f"{b0['sdp_row_ratio']:.2f}/{b1['sdp_row_ratio']:.2f}")

    L.append("\n## 5) summary_by_subtype_by_candidate.csv 핵심행 → 파일 참조")
    return "\n".join(L) + "\n"


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="게이트 1-b: beep concat artifact 보강 진단 (read-only)")
    ap.add_argument("--ckpt", action="append", type=Path, default=None,
                    help="6-class pretrained6 계열 checkpoint. 여러 번 지정 가능. 미지정 시 기본 후보 자동.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-model", action="store_true", help="모델 비교 생략(SDP-only)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT / "plots"
    arrays_dir = OUT / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    # ── checkpoints ──────────────────────────────────────────────────────────
    checkpoints, ckpt_infos, ckpt_ids = [], [], []
    have_model = not args.no_model
    device = None
    if have_model:
        import torch
        device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  if args.device == "auto" else torch.device(args.device))
        if args.ckpt:
            ck_specs = [(f"{p.parent.name}__{p.stem}", p) for p in args.ckpt]
        else:
            ck_specs = default_checkpoints()
        if not ck_specs:
            print("[WARN] checkpoint 없음 → SDP-only 로 진행")
            have_model = False
        for ck_id, p in ck_specs:
            if not p.exists():
                print(f"[WARN] ckpt 없음 skip: {p}")
                continue
            model, info, supp = load_pretrained6_model(p, device)
            info["checkpoint_id"] = ck_id
            checkpoints.append((ck_id, model, supp))
            ckpt_infos.append(info)
            ckpt_ids.append(ck_id)
            print(f"[ckpt] {ck_id}: out_dim={info['output_dim']} classes={info['classes']} "
                  f"attn={supp} device={device}")
        if not checkpoints:
            have_model = False
    attn_available = any(supp for _, _, supp in checkpoints)

    if not HAVE_MPL:
        print(f"[WARN] matplotlib 미사용 ({_MPL_ERR}) → PNG 생략")

    # ── 세션 선택 ────────────────────────────────────────────────────────────
    print("\n=== 세션 선택 ===")
    primary_sessions, primary_records = select_primary()
    primary_paths = {str(s["path"]) for s in primary_sessions}
    walk_sessions, walk_records = select_walk_cross_subject(primary_paths)
    sel_records = primary_records + walk_records
    write_csv(OUT / "selected_sessions.csv", sel_records,
              ["file", "subtype", "env", "subject", "trial", "resampled_count",
               "selected", "selection_group", "skip_reason"])
    sessions = primary_sessions + walk_sessions
    print(f"\n분석 대상: primary {len(primary_sessions)} + walk-cross {len(walk_sessions)} = {len(sessions)}")

    # ── 분석 ──────────────────────────────────────────────────────────────────
    print("\n=== 분석 ===")
    do_plots = HAVE_MPL and not args.no_plots
    if do_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    consistency_box = {"done": False, "ok": None}
    records = []
    for sess in sessions:
        rec = analyze_session(sess, checkpoints, device, consistency_box)
        b0, b1 = rec["boundary"][50], rec["boundary"][250]
        msg = (f"[an] {rec['subtype']:12s} E{rec['env']}_S{rec['subject']:02d}_T{rec['trial']:03d} "
               f"({rec['selection_group']:17s}) amp {b0['amp_jump_ratio']:.2f}/{b1['amp_jump_ratio']:.2f} "
               f"sdp_row {b0['sdp_row_ratio']:.2f}/{b1['sdp_row_ratio']:.2f}")
        if have_model:
            ck0 = ckpt_ids[0]
            mc = rec["model_by_ckpt"][ck0]
            msg += (f" | [{ck0}] fall cm={mc['concat_main']['fall_prob']:.3f} "
                    f"ctr={mc['continuous_center']['fall_prob']:.3f}")
        print(msg)

        # npz
        arr = rec["_arrays"]
        npz = {f"sdp_z__{n}": arr["sdp_z"][n] for n in ALL_CROPS}
        npz["sparse_frame_energy_concat"] = arr["sparse_frame_energy_concat"]
        npz["sdp_row_energy_concat_z"] = arr["sdp_row_energy_concat_z"]
        if have_model:
            for ck_id in ckpt_ids:
                for n in ALL_CROPS:
                    npz[f"prob__{ck_id}__{n}"] = arr["probs"][ck_id][n]
                    if arr["attn"][ck_id][n] is not None:
                        npz[f"attn__{ck_id}__{n}"] = arr["attn"][ck_id][n]
        stem = f"{rec['subtype']}__E{rec['env']}_S{rec['subject']:02d}_T{rec['trial']:03d}"
        np.savez_compressed(arrays_dir / f"{stem}.npz", **npz)

        if do_plots:
            plot_session(rec, plots_dir, ckpt_ids, have_model)
        rec.pop("_arrays", None)
        records.append(rec)

    # ── 산출물 ────────────────────────────────────────────────────────────────
    metric_rows = build_metric_rows(records, ckpt_ids, have_model)
    per_cand_header = [
        "file", "subtype", "env", "subject", "trial", "selection_group",
        "checkpoint_id", "model_available", "candidate", "coordinate_policy",
        "raw_sdp_cosine", "raw_sdp_mad", "z_sdp_cosine", "z_sdp_mad",
        "energy_curve_corr", "sdp_peak_row_diff", "sparse_peak_frame_diff",
    ]
    if have_model:
        per_cand_header += [
            "fall_prob", "pred_class", "fall_rank", "max_nonfall_prob", "fall_margin",
            "crosses_thr_0p1", "softmax_l1_diff_vs_concat",
            "fall_prob_diff_concat_minus_candidate", "concat_crosses_but_candidate_not",
            "attention_available", "attention_boundary_mass", "attention_total_mass",
            "attention_boundary_ratio", "attention_peak_row", "attention_peak_value",
        ]
    write_csv(OUT / "per_candidate_metrics.csv", round_rows(metric_rows), per_cand_header)

    # boundary (concat_main, ckpt-독립)
    bnd_rows = []
    for r in records:
        for b in BOUNDARIES:
            d = r["boundary"][b]
            bnd_rows.append({
                "file": r["file"], "subtype": r["subtype"], "env": r["env"], "subject": r["subject"],
                "trial": r["trial"], "selection_group": r["selection_group"], "boundary_b": b,
                "amp_jump": d["amp_jump"], "amp_nonbnd_median": d["amp_nonbnd_median"],
                "amp_jump_ratio": d["amp_jump_ratio"],
                "sparse_be": d["sparse_be"], "sparse_nonbnd_median": d["sparse_nonbnd_median"],
                "sparse_be_ratio": d["sparse_be_ratio"],
                "sdp_boundary_rows": "|".join(map(str, d["sdp_boundary_rows"])),
                "sdp_boundary_rows_energy_median": d["sdp_boundary_rows_energy_median"],
                "sdp_nonbnd_rows_median": d["sdp_nonbnd_rows_median"], "sdp_row_ratio": d["sdp_row_ratio"],
            })
    write_csv(OUT / "boundary_artifact_metrics.csv", round_rows(bnd_rows),
              ["file", "subtype", "env", "subject", "trial", "selection_group", "boundary_b",
               "amp_jump", "amp_nonbnd_median", "amp_jump_ratio", "sparse_be", "sparse_nonbnd_median",
               "sparse_be_ratio", "sdp_boundary_rows", "sdp_boundary_rows_energy_median",
               "sdp_nonbnd_rows_median", "sdp_row_ratio"])

    agg = aggregate_by_subtype_by_candidate(metric_rows, have_model)
    agg_header = ["subtype", "candidate", "checkpoint_id", "model_available", "n_sessions",
                  "z_sdp_cosine_median", "z_sdp_cosine_min", "raw_sdp_cosine_median", "raw_sdp_cosine_min",
                  "energy_curve_corr_median", "energy_curve_corr_min",
                  "sdp_peak_row_diff_median", "sdp_peak_row_diff_max",
                  "sparse_peak_frame_diff_median", "sparse_peak_frame_diff_max"]
    if have_model:
        agg_header += ["fall_prob_median", "fall_prob_max",
                       "fall_prob_diff_concat_minus_candidate_median", "fall_prob_diff_concat_minus_candidate_max",
                       "crosses_thr_0p1_count", "concat_crosses_but_candidate_not_count",
                       "attention_boundary_mass_median", "attention_boundary_mass_max",
                       "attention_boundary_ratio_median", "attention_boundary_ratio_max"]
    write_csv(OUT / "summary_by_subtype_by_candidate.csv", round_rows(agg), agg_header)

    share = build_share_summary(records, agg, ckpt_ids, have_model, attn_available)
    (OUT / "share_summary.md").write_text(share, encoding="utf-8")

    summary = {
        "meta": {
            "purpose": "Gate 1-b — beep concat artifact 모델 레벨 보강 진단 (read-only)",
            "note": "continuous 후보는 concat_main 과 동일 내용 1:1 대응 아님 (beep 미제거 대체 crop).",
            "coordinate_policy_kept": ["clean400_concat", "continuous_center"],
            "thr_helper": THR,
            "preprocess": {"rpca_max_iter": DEFAULT_MAX_ITER, "rpca_tol": None,
                           "sdp_sub_w": SUB_W, "sdp_stride": SUB_STRIDE, "n_lags": N_LAGS},
            "boundaries_crop_local": BOUNDARIES, "boundary_rows": BOUNDARY_ROWS,
            "continuous_candidates": {k: list(v) for k, v in CONTINUOUS.items()},
            "checkpoints": ckpt_infos, "have_model": have_model, "attn_available": attn_available,
            "window_to_model_input_consistency": consistency_box["ok"],
            "n_sessions": len(records),
        },
        "selected_sessions": sel_records,
        "aggregate_by_subtype_by_candidate": agg,
        "sessions": records,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 산출 완료 ===")
    for f in ["selected_sessions.csv", "per_candidate_metrics.csv", "boundary_artifact_metrics.csv",
              "summary_by_subtype_by_candidate.csv", "share_summary.md", "summary.json"]:
        print(f"  {OUT / f}")
    print(f"  {arrays_dir}\\*.npz" + (f"  /  {plots_dir}\\*.png" if do_plots else ""))
    print(f"\n[consistency] window_to_model_input == preprocess_crop : {consistency_box['ok']}")
    print("\n----- share_summary.md -----")
    print(share)


if __name__ == "__main__":
    main()
