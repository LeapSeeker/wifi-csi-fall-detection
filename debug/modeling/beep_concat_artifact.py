"""beep 제거 concat 접합부 artifact 검증 (read-only).

절차(프롬프트 확정):
  resample_to_100hz 후 n_frames in [550,555]만 사용 (n<550 제외, n>550 tail 별도기록).
  beep frame: Stage1[0:50], Stage2[150:200], Stage3[400:450] (first 550 기준).
  clean = concat(A[50:150], A[200:400], A[450:550]) -> 400 frames.
    junction: clean idx 100 (pre->fall), 300 (fall->post).
  [2] frame-to-frame L2 delta (전체 104dim), junction vs non-junction p50/p95/p99.
  [3] Y = clean[50:350] (300f) -> RPCA->ACF->SDP (28,20).
      junction clean100/300 -> Y-local 50/250 -> SDP bins {3,4,5}/{23,24,25}.
      bin energy = mean(abs(SDP_z[row,:])), junction-bin energy vs median 배수.
  [4] 연속 crop Z1=150:450, Z2=200:500, Z3=100:400 (pad 없음) SDP 안정성/fall포함.
출력: stdout + diag_out/beep_artifact/{summary.csv,summary.json}. 원본 read-only.
"""
from __future__ import annotations
import json
import sys
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model.preprocessing.loader import load_safesignal_csv  # noqa: E402
from model.preprocessing.resample import resample_to_100hz  # noqa: E402
from model.preprocessing.rpca import rpca_sparse  # noqa: E402
from model.preprocessing.sdp import stacked_doppler_profile, SUB_W, SUB_STRIDE  # noqa: E402

OUT = ROOT / "debug/modeling/diag_out/beep_artifact"
CLEANED = ROOT / "data/cleaned"
SUBTYPES = ["FALL_WALK_F", "FALL_WALK_B", "FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B"]
PER_SUBTYPE = 2
BEEPS = {"Stage1": (0, 50), "Stage2": (150, 200), "Stage3": (400, 450)}
FALL_REGION = (200, 400)  # clean fall segment 원본 인덱스


def sdp_bins_for_frame(f, sub_w=SUB_W, stride=SUB_STRIDE, n_bins=28):
    lo = int(np.ceil((f - (sub_w - 1)) / stride))
    hi = int(np.floor(f / stride))
    return [i for i in range(max(0, lo), min(n_bins - 1, hi) + 1)]


def zscore(sdp):
    return (sdp - sdp.mean()) / (sdp.std() + 1e-6)


def bin_energy(sdp_z):
    return np.abs(sdp_z).mean(axis=1)  # (28,)


def analyze_session(path):
    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    A = res.amplitude
    n = int(A.shape[0])
    dim = int(A.shape[1])
    if n < 550:
        return {"file": Path(path).name, "n_frames": n, "dim": dim, "status": "EXCLUDED_<550"}
    tail = n - 550
    A = A[:550]
    clean = np.concatenate([A[50:150], A[200:400], A[450:550]], axis=0)  # (400, dim)

    # [2] junction delta
    d = np.linalg.norm(np.diff(clean, axis=0), axis=1)  # (399,) d[i]=||clean[i+1]-clean[i]||
    # junction crossing splice: between clean[99],clean[100] -> d[99]; clean[299],clean[300] -> d[299]
    jidx = [99, 299]
    non_j = np.delete(d, jidx)
    p50, p95, p99 = np.percentile(non_j, [50, 95, 99])
    jd = {"pre_fall": float(d[99]), "fall_post": float(d[299])}
    delta_block = {
        "junction_delta": jd,
        "nonjunction_p50": float(p50), "nonjunction_p95": float(p95), "nonjunction_p99": float(p99),
        "pre_fall_gt_p95": bool(d[99] > p95), "pre_fall_gt_p99": bool(d[99] > p99),
        "fall_post_gt_p95": bool(d[299] > p95), "fall_post_gt_p99": bool(d[299] > p99),
        "pre_fall_ratio_p99": float(d[99] / p99) if p99 else None,
        "fall_post_ratio_p99": float(d[299] / p99) if p99 else None,
    }

    # [3] Y SDP
    Y = clean[50:350]  # (300, dim)
    sdpY = zscore(stacked_doppler_profile(rpca_sparse(Y)))
    eY = bin_energy(sdpY)
    medY = float(np.median(eY))
    binsA = sdp_bins_for_frame(50)   # {3,4,5}
    binsB = sdp_bins_for_frame(250)  # {23,24,25}
    sdp_block = {
        "junction_local_frames": {"pre_fall": 50, "fall_post": 250},
        "junction_bins": {"pre_fall": binsA, "fall_post": binsB},
        "bin_energy_median": medY,
        "pre_fall_bins_energy": [float(eY[i]) for i in binsA],
        "fall_post_bins_energy": [float(eY[i]) for i in binsB],
        "pre_fall_maxratio_median": float(max(eY[i] for i in binsA) / medY) if medY else None,
        "fall_post_maxratio_median": float(max(eY[i] for i in binsB) / medY) if medY else None,
    }
    # ACF lag spike: per junction bin set, max |val| lag vs median across bins at that lag
    sdpY_abs = np.abs(sdpY)  # (28,20)
    lag_med = np.median(sdpY_abs, axis=0)  # (20,) median over bins per lag
    def lag_spike(bins):
        best = {"ratio": 0.0, "lag": None, "bin": None}
        for b in bins:
            for lag in range(sdpY_abs.shape[1]):
                r = sdpY_abs[b, lag] / (lag_med[lag] + 1e-9)
                if r > best["ratio"]:
                    best = {"ratio": float(r), "lag": int(lag + 1), "bin": int(b)}
        return best
    sdp_block["acf_lag_spike_pre_fall"] = lag_spike(binsA)
    sdp_block["acf_lag_spike_fall_post"] = lag_spike(binsB)

    # [4] continuous crops (pad 없음)
    def beeps_in(lo, hi):
        return [name for name, (a, b) in BEEPS.items() if a >= lo and b <= hi or (a < hi and b > lo)]
    crops = {"Z1": (150, 450), "Z2": (200, 500), "Z3": (100, 400), "X": (150, 450)}
    crop_block = {}
    for name, (lo, hi) in crops.items():
        C = A[lo:hi]
        sdpC = zscore(stacked_doppler_profile(rpca_sparse(C)))
        eC = bin_energy(sdpC)
        fa, fb = FALL_REGION
        ov = max(0, min(hi, fb) - max(lo, fa))
        fall_cov = ov / (fb - fa)
        fall_center_local = (fa + fb) / 2 - lo
        beep_overlap = [name2 for name2, (a, b) in BEEPS.items() if min(hi, b) - max(lo, a) > 0]
        crop_block[name] = {
            "range": [lo, hi],
            "energy_mean": float(eC.mean()), "energy_cv": float(eC.std() / (eC.mean() + 1e-9)),
            "energy_max": float(eC.max()), "energy_peak_bin": int(eC.argmax()),
            "fall_coverage": float(fall_cov), "fall_center_local": float(fall_center_local),
            "beeps_contained": beep_overlap,
        }

    return {"file": Path(path).name, "n_frames": n, "dim": dim, "tail_over550": int(tail),
            "status": "OK", "delta": delta_block, "sdp": sdp_block, "crops": crop_block}


def main():
    selected = []
    for st in SUBTYPES:
        files = sorted(CLEANED.rglob(f"*{st}*_T*.csv"))
        got = 0
        for f in files:
            r = analyze_session(f)
            if r["status"] == "OK":
                r["subtype"] = st
                selected.append(r)
                print(f"[OK] {st} {r['file']} n={r['n_frames']} tail={r['tail_over550']}")
                got += 1
                if got >= PER_SUBTYPE:
                    break
            else:
                print(f"[skip] {st} {Path(f).name}: {r['status']} (n={r.get('n_frames')})")
        if got == 0:
            print(f"[WARN] {st}: 정상길이(550-555) 세션 없음")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "beep_artifact_summary.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    # CSV (flat)
    with open(OUT / "beep_artifact_summary.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["subtype", "file", "n_frames", "tail", "d_pre", "d_post", "p95", "p99",
                    "pre>p99", "post>p99", "pre_bin_ratio", "post_bin_ratio",
                    "pre_acf_ratio", "post_acf_ratio"])
        for r in selected:
            dl, sd = r["delta"], r["sdp"]
            w.writerow([r["subtype"], r["file"], r["n_frames"], r["tail_over550"],
                        round(dl["junction_delta"]["pre_fall"], 3), round(dl["junction_delta"]["fall_post"], 3),
                        round(dl["nonjunction_p95"], 3), round(dl["nonjunction_p99"], 3),
                        dl["pre_fall_gt_p99"], dl["fall_post_gt_p99"],
                        round(sd["pre_fall_maxratio_median"], 2), round(sd["fall_post_maxratio_median"], 2),
                        round(sd["acf_lag_spike_pre_fall"]["ratio"], 2), round(sd["acf_lag_spike_fall_post"]["ratio"], 2)])
    print("\n===JSON_SUMMARY_BEGIN===")
    print(json.dumps(selected, indent=2, ensure_ascii=False))
    print("===JSON_SUMMARY_END===")


if __name__ == "__main__":
    main()
