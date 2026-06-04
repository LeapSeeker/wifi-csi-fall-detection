"""beep-in-crop SDP/모델 영향 측정 — concat(Y) vs 연속 Z1(150:450) 비교 (진단용, read-only).

목적
----
event-centered 절단 방식 후보 비교:
  Y(concat 4초 복원): beep 제거 clean 4초에서 300f 절단 (beep 없음)
  Z1(연속 150:450): splice 없음, Stage2/Stage3 beep 2개 포함

핵심 질문: Z1의 beep이 (a) fall transient를 가리는지, (b) standing-like 정지 단서를
강화하는지 진단.

주의: 진단용. 현재 checkpoint는 기존 sliding/weak-label cache 학습 모델이므로
probability 비교를 "Z1/concat 최종 성능"으로 단정하지 말 것.

원본/cache/checkpoint/report 수정 금지. 학습/git 금지. 추론만.
산출물: debug/modeling/diag_out/beep_incrop/{beep_incrop_summary.csv,beep_incrop_summary.json}
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

from model.preprocessing.loader import load_safesignal_csv          # noqa: E402
from model.preprocessing.resample import resample_to_100hz          # noqa: E402
from model.preprocessing.rpca import rpca_sparse                    # noqa: E402
from model.preprocessing.sdp import stacked_doppler_profile, SUB_W, SUB_STRIDE, W_T  # noqa: E402
from model.preprocessing.pipeline import window_to_model_input      # noqa: E402
from model.finetune.train import PRETRAINED6_CLASSES                # noqa: E402
from model.pretrained.model import CNNGRUAttention                  # noqa: E402

OUT = ROOT / "debug/modeling/diag_out/beep_incrop"
CLEANED = ROOT / "data/cleaned"
# 직전 검증(beep_concat_artifact.py)과 동일 선택: subtype별 2개, FALL_WALK_B 포함.
SUBTYPES = ["FALL_WALK_F", "FALL_WALK_B", "FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B"]
PER_SUBTYPE = 2

CKPT_PRIMARY = ROOT / "model/finetune/checkpoints_track1_formal_global_fw15_s43"
CKPT_FALLBACK_GLOB = "checkpoints_track1_formal_global_fw15_s*"

FALL_IDX = PRETRAINED6_CLASSES.index("fall")        # 0
STANDING_IDX = PRETRAINED6_CLASSES.index("standing")  # 4
N_BINS = W_T  # 28

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── [1] bin 매핑 ────────────────────────────────────────────────────────────
def sdp_bins_for_frame(f, sub_w=SUB_W, stride=SUB_STRIDE, n_bins=N_BINS):
    """frame f 를 포함하는 SDP bin 집합. bin i = frame [stride*i, stride*i+sub_w).
    i in [ceil((f-(sub_w-1))/stride), floor(f/stride)] ∩ [0, n_bins-1]."""
    lo = int(np.ceil((f - (sub_w - 1)) / stride))
    hi = int(np.floor(f / stride))
    return [i for i in range(max(0, lo), min(n_bins - 1, hi) + 1)]


def bins_for_range(f_lo, f_hi):
    """[f_lo, f_hi] (inclusive) 프레임을 포함하는 bin 합집합."""
    s = set()
    for f in range(f_lo, f_hi + 1):
        s.update(sdp_bins_for_frame(f))
    return sorted(s)


FALL_ONSET_BINS = sdp_bins_for_frame(50)          # local frame 50 → onset
FALL_RANGE_BINS = bins_for_range(50, 250)         # local 50~250
BEEP_S2_BINS = bins_for_range(0, 49)              # Z1 Stage2 beep local [0:50]
BEEP_S3_BINS = bins_for_range(250, 299)           # Z1 Stage3 beep local [250:300]


# ─── crop 구성 ───────────────────────────────────────────────────────────────
def build_Y(A550):
    clean = np.concatenate([A550[50:150], A550[200:400], A550[450:550]], axis=0)  # (400, dim)
    return clean[50:350]  # (300, dim)


def build_Z1(A550):
    return A550[150:450]  # (300, dim)


# ─── SDP 헬퍼 ────────────────────────────────────────────────────────────────
def raw_sdp(window):
    """pre-zscore SDP: RPCA sparse → stacked_doppler_profile. (28,20)."""
    return stacked_doppler_profile(rpca_sparse(window))


def zscore(sdp):
    return (sdp - sdp.mean()) / (sdp.std() + 1e-6)


def bin_energy(sdp):
    """bin별 energy = mean(abs(SDP[row, :]))  → (28,)."""
    return np.abs(sdp).mean(axis=1)


def mean_bins(energy, bins):
    return float(np.mean([energy[i] for i in bins]))


# ─── 모델 ────────────────────────────────────────────────────────────────────
def resolve_ckpt():
    if (CKPT_PRIMARY / "best_operating.pt").exists():
        return CKPT_PRIMARY
    cands = sorted((ROOT / "model/finetune").glob(CKPT_FALLBACK_GLOB))
    for c in cands:
        if (c / "best_operating.pt").exists():
            return c
    raise FileNotFoundError("fw1.5 checkpoint (best_operating.pt) not found")


def load_model(ckpt_dir):
    ck = torch.load(ckpt_dir / "best_operating.pt", map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m = CNNGRUAttention(n_classes=len(PRETRAINED6_CLASSES)).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, ck


def infer_probs(model, model_input):
    """model_input: (1,1,28,20) np.float32 → softmax probs (6,)."""
    xb = torch.as_tensor(model_input, dtype=torch.float32).to(device)
    with torch.no_grad():
        p = torch.softmax(model(xb), dim=1).cpu().numpy()[0]
    return p


# ─── 세션 분석 ───────────────────────────────────────────────────────────────
def analyze_session(path, model):
    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    A = res.amplitude
    n = int(A.shape[0])
    dim = int(A.shape[1])
    if not (550 <= n <= 555):
        return {"file": Path(path).name, "n_frames": n, "dim": dim,
                "status": f"EXCLUDED(n={n} not in 550..555)"}
    A550 = A[:550]

    out = {"file": Path(path).name, "n_frames": n, "dim": dim, "status": "OK"}

    windows = {"Y": build_Y(A550), "Z1": build_Z1(A550)}

    # [2] pre-zscore + [3] post-zscore/model-input
    pre, post, minput, probs = {}, {}, {}, {}
    for name, w in windows.items():
        sdp_raw = raw_sdp(w)                 # pre-zscore (28,20)
        sdp_z = zscore(sdp_raw)              # post-zscore (28,20)
        mi = window_to_model_input(w)        # 계약: raw (300,dim) → (1,28,20), 내부 RPCA→SDP→zscore
        # cross-check: window_to_model_input 결과가 우리의 zscore(raw_sdp)와 일치 → 이중 RPCA/zscore 없음 검증
        consistent = bool(np.allclose(mi[0], sdp_z, atol=1e-4))
        pre[name] = sdp_raw
        post[name] = sdp_z
        minput[name] = mi
        probs[name] = infer_probs(model, mi[None, ...])  # (1,1,28,20)
        out.setdefault("_consistency", {})[name] = consistent
        out.setdefault("_minput_shape", {})[name] = list(mi.shape)
        out.setdefault("_minput_range", {})[name] = [float(mi.min()), float(mi.max())]
        out.setdefault("_raw_sdp_range", {})[name] = [float(sdp_raw.min()), float(sdp_raw.max())]

    eY_pre, eZ_pre = bin_energy(pre["Y"]), bin_energy(pre["Z1"])
    eY_post, eZ_post = bin_energy(post["Y"]), bin_energy(post["Z1"])

    # [2] pre-zscore 블록
    z_beep_s2 = mean_bins(eZ_pre, BEEP_S2_BINS)
    z_beep_s3 = mean_bins(eZ_pre, BEEP_S3_BINS)
    z_med = float(np.median(eZ_pre))
    out["pre_zscore"] = {
        "energy_def": "mean(abs(SDP_raw[row,:]))",
        "Y_fall_onset": mean_bins(eY_pre, FALL_ONSET_BINS),
        "Z1_fall_onset": mean_bins(eZ_pre, FALL_ONSET_BINS),
        "Y_fall_range": mean_bins(eY_pre, FALL_RANGE_BINS),
        "Z1_fall_range": mean_bins(eZ_pre, FALL_RANGE_BINS),
        "Z1_beep_s2": z_beep_s2,
        "Z1_beep_s3": z_beep_s3,
        "Z1_bin_energy_median": z_med,
        # beep 구간이 저변화·평탄 정지 패턴(중앙값 이하)인가
        "Z1_beep_s2_below_median": bool(z_beep_s2 < z_med),
        "Z1_beep_s3_below_median": bool(z_beep_s3 < z_med),
        "Z1_beep_s2_ratio_median": z_beep_s2 / z_med if z_med else None,
        "Z1_beep_s3_ratio_median": z_beep_s3 / z_med if z_med else None,
        # beep이 fall onset energy를 (전체 평탄화로) 희석하는지: Z1 fall_onset / Y fall_onset
        "fall_onset_Z1_over_Y": (mean_bins(eZ_pre, FALL_ONSET_BINS)
                                 / mean_bins(eY_pre, FALL_ONSET_BINS)) if mean_bins(eY_pre, FALL_ONSET_BINS) else None,
        "fall_range_Z1_over_Y": (mean_bins(eZ_pre, FALL_RANGE_BINS)
                                 / mean_bins(eY_pre, FALL_RANGE_BINS)) if mean_bins(eY_pre, FALL_RANGE_BINS) else None,
    }

    # [3] post-zscore / model-input 블록
    out["post_zscore"] = {
        "Y_fall_onset": mean_bins(eY_post, FALL_ONSET_BINS),
        "Z1_fall_onset": mean_bins(eZ_post, FALL_ONSET_BINS),
        "Y_fall_range": mean_bins(eY_post, FALL_RANGE_BINS),
        "Z1_fall_range": mean_bins(eZ_post, FALL_RANGE_BINS),
        "Z1_beep_s2": mean_bins(eZ_post, BEEP_S2_BINS),
        "Z1_beep_s3": mean_bins(eZ_post, BEEP_S3_BINS),
        "fall_onset_Z1_minus_Y": mean_bins(eZ_post, FALL_ONSET_BINS) - mean_bins(eY_post, FALL_ONSET_BINS),
        "fall_range_Z1_minus_Y": mean_bins(eZ_post, FALL_RANGE_BINS) - mean_bins(eY_post, FALL_RANGE_BINS),
    }

    # [4] 모델 probability
    out["prob"] = {
        "fall_Y": float(probs["Y"][FALL_IDX]),
        "fall_Z1": float(probs["Z1"][FALL_IDX]),
        "standing_Y": float(probs["Y"][STANDING_IDX]),
        "standing_Z1": float(probs["Z1"][STANDING_IDX]),
        "fall_drop_Z1": float(probs["Z1"][FALL_IDX] - probs["Y"][FALL_IDX]),       # <0 이면 하락
        "standing_rise_Z1": float(probs["Z1"][STANDING_IDX] - probs["Y"][STANDING_IDX]),  # >0 이면 상승
        "argmax_Y": PRETRAINED6_CLASSES[int(probs["Y"].argmax())],
        "argmax_Z1": PRETRAINED6_CLASSES[int(probs["Z1"].argmax())],
    }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = resolve_ckpt()
    model, ck = load_model(ckpt_dir)
    print(f"[ckpt] {ckpt_dir.name}/best_operating.pt  saved_thr={ck.get('threshold')}")
    print(f"[bins] fall_onset={FALL_ONSET_BINS} fall_range={FALL_RANGE_BINS}")
    print(f"[bins] beep_s2(Z1)={BEEP_S2_BINS} beep_s3(Z1)={BEEP_S3_BINS}")
    print(f"[bins] overlap fall_onset∩beep_s2={sorted(set(FALL_ONSET_BINS)&set(BEEP_S2_BINS))} "
          f"fall_range∩beep_s3={sorted(set(FALL_RANGE_BINS)&set(BEEP_S3_BINS))}")

    selected = []
    for st in SUBTYPES:
        files = sorted(CLEANED.rglob(f"*{st}*_T*.csv"))
        got = 0
        for f in files:
            r = analyze_session(f, model)
            if r["status"] == "OK":
                r["subtype"] = st
                selected.append(r)
                p = r["prob"]
                print(f"[OK] {st} {r['file']} n={r['n_frames']} dim={r['dim']} | "
                      f"fall Y={p['fall_Y']:.3f} Z1={p['fall_Z1']:.3f} | "
                      f"stand Y={p['standing_Y']:.3f} Z1={p['standing_Z1']:.3f} | "
                      f"consist={r['_consistency']}")
                got += 1
                if got >= PER_SUBTYPE:
                    break
            else:
                print(f"[skip] {st} {Path(f).name}: {r['status']}")
        if got == 0:
            print(f"[WARN] {st}: 정상길이(550-555) 세션 없음")

    # ─── 집계 ────────────────────────────────────────────────────────────────
    def col(key_path):
        vals = []
        for r in selected:
            d = r
            for k in key_path:
                d = d[k]
            vals.append(d)
        return np.array(vals, dtype=float)

    agg = {}
    if selected:
        fall_drop = col(["prob", "fall_drop_Z1"])
        stand_rise = col(["prob", "standing_rise_Z1"])
        agg = {
            "n_sessions": len(selected),
            "fall_Y_mean_std": [float(col(["prob", "fall_Y"]).mean()), float(col(["prob", "fall_Y"]).std())],
            "fall_Z1_mean_std": [float(col(["prob", "fall_Z1"]).mean()), float(col(["prob", "fall_Z1"]).std())],
            "standing_Y_mean_std": [float(col(["prob", "standing_Y"]).mean()), float(col(["prob", "standing_Y"]).std())],
            "standing_Z1_mean_std": [float(col(["prob", "standing_Z1"]).mean()), float(col(["prob", "standing_Z1"]).std())],
            "n_Z1_fall_drop": int((fall_drop < 0).sum()),
            "n_Z1_standing_rise": int((stand_rise > 0).sum()),
            "fall_drop_mean": float(fall_drop.mean()),
            "standing_rise_mean": float(stand_rise.mean()),
            "pre_fall_onset_Z1_over_Y_mean": float(col(["pre_zscore", "fall_onset_Z1_over_Y"]).mean()),
            "pre_fall_range_Z1_over_Y_mean": float(col(["pre_zscore", "fall_range_Z1_over_Y"]).mean()),
            "post_fall_onset_Z1_minus_Y_mean": float(col(["post_zscore", "fall_onset_Z1_minus_Y"]).mean()),
            "post_fall_range_Z1_minus_Y_mean": float(col(["post_zscore", "fall_range_Z1_minus_Y"]).mean()),
            "n_Z1_beep_s2_below_median": int(sum(r["pre_zscore"]["Z1_beep_s2_below_median"] for r in selected)),
            "n_Z1_beep_s3_below_median": int(sum(r["pre_zscore"]["Z1_beep_s3_below_median"] for r in selected)),
            "all_consistent": all(all(r["_consistency"].values()) for r in selected),
        }

    payload = {
        "meta": {
            "checkpoint": ckpt_dir.name,
            "saved_threshold": ck.get("threshold"),
            "classes": list(PRETRAINED6_CLASSES),
            "fall_idx": FALL_IDX, "standing_idx": STANDING_IDX,
            "bins": {
                "fall_onset": FALL_ONSET_BINS, "fall_range": FALL_RANGE_BINS,
                "beep_s2_Z1": BEEP_S2_BINS, "beep_s3_Z1": BEEP_S3_BINS,
                "overlap_onset_beep_s2": sorted(set(FALL_ONSET_BINS) & set(BEEP_S2_BINS)),
                "overlap_range_beep_s3": sorted(set(FALL_RANGE_BINS) & set(BEEP_S3_BINS)),
            },
            "device": str(device),
            "note": "진단용. 현재 checkpoint는 기존 sliding/weak-label cache 학습 모델.",
        },
        "aggregate": agg,
        "sessions": selected,
    }
    (OUT / "beep_incrop_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV (flat per-session)
    with open(OUT / "beep_incrop_summary.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "subtype", "file", "n_frames", "dim",
            "pre_Y_onset", "pre_Z1_onset", "pre_Y_range", "pre_Z1_range",
            "pre_Z1_beep_s2", "pre_Z1_beep_s3", "pre_Z1_med",
            "pre_onset_Z1/Y", "pre_range_Z1/Y",
            "post_Y_onset", "post_Z1_onset", "post_onset_Z1-Y",
            "fall_Y", "fall_Z1", "fall_drop", "stand_Y", "stand_Z1", "stand_rise",
            "argmax_Y", "argmax_Z1", "consistent",
        ])
        for r in selected:
            pz, po, pr = r["pre_zscore"], r["post_zscore"], r["prob"]
            w.writerow([
                r["subtype"], r["file"], r["n_frames"], r["dim"],
                round(pz["Y_fall_onset"], 4), round(pz["Z1_fall_onset"], 4),
                round(pz["Y_fall_range"], 4), round(pz["Z1_fall_range"], 4),
                round(pz["Z1_beep_s2"], 4), round(pz["Z1_beep_s3"], 4), round(pz["Z1_bin_energy_median"], 4),
                round(pz["fall_onset_Z1_over_Y"], 3) if pz["fall_onset_Z1_over_Y"] else None,
                round(pz["fall_range_Z1_over_Y"], 3) if pz["fall_range_Z1_over_Y"] else None,
                round(po["Y_fall_onset"], 4), round(po["Z1_fall_onset"], 4), round(po["fall_onset_Z1_minus_Y"], 4),
                round(pr["fall_Y"], 4), round(pr["fall_Z1"], 4), round(pr["fall_drop_Z1"], 4),
                round(pr["standing_Y"], 4), round(pr["standing_Z1"], 4), round(pr["standing_rise_Z1"], 4),
                pr["argmax_Y"], pr["argmax_Z1"], all(r["_consistency"].values()),
            ])

    print("\n===AGG===")
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"\n[out] {OUT/'beep_incrop_summary.csv'}")
    print(f"[out] {OUT/'beep_incrop_summary.json'}")


if __name__ == "__main__":
    main()
