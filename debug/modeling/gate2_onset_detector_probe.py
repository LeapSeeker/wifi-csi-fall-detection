"""게이트 2: onset detector probe (read-only).

전체 fall 세션에 대해 onset 자동 후보·baseline 품질·confidence 구성요소·needs_review
사유를 산출한다. 이번 probe는 search 범위·k/M·soft threshold를 확정하지 않는다.
여러 후보(grid)를 산출한 뒤 train/val 분포를 보고 최종 detector 기준을 정한다.
window_manifest는 항목 4·5 crop/label 정책 확정 후 별도 생성(이번 범위 아님).

좌표계 (게이트 1 확정):
  original = 100Hz resample 후 [:550]
  clean400 = original[50:150] + original[200:400] + original[450:550]
  nominal onset = original 200 = clean 100, nominal fall = original[200:400] = clean[100:300]
  original→clean: [50:150]→[0:100], [200:400]→[100:300], [450:550]→[300:400]
  beep(직접매핑 안 함): original[0:50] stage1 / [150:200] stage2 / [400:450] stage3
  onset detector는 full original 550 좌표계에서 수행(clean crop 내부 아님).

검출:
  rise signal = full 550 RPCA sparse frame energy → 5-frame smoothing
  rise = baseline_median + k_mad*baseline_mad 초과가 sustain_frames 지속되는 첫 지점
  peak = search 구간 내 rise 이후 smoothed energy 최대 (절대 height threshold 미사용)
  grid: smooth=5 / k_mad∈{2.5,3.0,3.5} / sustain∈{3,5} / search∈{nominal,broad} → 세션당 12 row

split (D-031): split_id=within_subject_seed42_val0.2_test0.2_pretrained6,
  split_source=split_safesignal_within_subject, split_unit=filename, test_sealed=true,
  thresholds_decided_from=train+val. cache row index split 재사용 금지 — filename 단위 재현.
  split 재현 cache에 없는 raw fall 파일 → split_assignment=out_of_scope, decision_scope=metadata_only.

제약: read-only, 데이터 미수정, 새 학습 금지, 동결 파일(pipeline/rpca/acf/sdp/학습·추론) 무수정.
      산출: debug/modeling/diag_out/onset_detector/ 하위.
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
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from model.preprocessing.loader import load_safesignal_csv, parse_safesignal_filename  # noqa: E402
from model.preprocessing.resample import resample_to_100hz                              # noqa: E402
from model.preprocessing.rpca import rpca_sparse, DEFAULT_MAX_ITER                      # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as _e:  # pragma: no cover
    HAVE_MPL = False
    _MPL_ERR = repr(_e)

OUT = ROOT / "debug/modeling/diag_out/onset_detector"
CLEANED = ROOT / "data/cleaned"
SPLIT_CACHE = ROOT / "model/finetune/cache/safesignal_e1234_pretrained6.npz"

MANIFEST_ID = "onset_probe_2026-06-05"
MANIFEST_VERSION = "probe_long_v1"
SPLIT_ID = "within_subject_seed42_val0.2_test0.2_pretrained6"
SPLIT_SOURCE = "split_safesignal_within_subject"
SPLIT_UNIT = "filename"
SPLIT_SCOPE = "demo_primary_pretrained6"
SEED, VAL_RATIO, TEST_RATIO = 42, 0.2, 0.2
SIDE_FALL_MARKERS = ("FALL_SIT_S", "FALL_STD_S", "FALL_WALK_S")

ORIGINAL_LEN = 550
BASELINE = (50, 150)            # original
SEARCH_NOMINAL = (190, 350)     # original
SEARCH_BROAD = (180, 400)
NOMINAL_ONSET_ORIG = 200
NOMINAL_ONSET_CLEAN = 100
FALL_ORIG = (200, 400)          # expected fall interval (peak 검사용)
BEEP_REGIONS = [(0, 50), (150, 200), (400, 450)]

SMOOTH = 5
K_MADS = [2.5, 3.0, 3.5]
SUSTAINS = [3, 5]
SLOPE_WINDOW = 5
TOPK = 3
TOPK_MULTIMODAL_SPREAD = 40     # probe 초기 flag (후속 조정 가능)
TOPK_NO_CONSENSUS_SPREAD = 80   # hard: top-k가 search 절반 이상 흩어짐(probe 초기, 조정 가능)
WEAK_SUBTYPES = {"FALL_WALK_B"}  # review_priority high (hard 아님)
EPS = 1e-9


# ─── original → clean frame 매핑 ─────────────────────────────────────────────
def to_clean(orig):
    if orig is None:
        return None
    if 50 <= orig < 150:
        return orig - 50
    if 200 <= orig < 400:
        return orig - 100
    if 450 <= orig < 550:
        return orig - 150
    return None  # beep 또는 범위 밖


def in_beep(orig):
    return any(a <= orig < b for a, b in BEEP_REGIONS)


# ─── 신호 ────────────────────────────────────────────────────────────────────
def frame_energy(mat):
    return np.abs(mat).mean(axis=1)


def smooth_centered(x, w=SMOOTH):
    n = len(x)
    h = w // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        a = max(0, i - h)
        b = min(n, i + h + 1)
        out[i] = x[a:b].mean()
    return out


def find_rise(es, s0, s1, thr, sustain):
    """[s0,s1) 에서 es[i:i+sustain] 가 모두 thr 초과인 첫 i. sustain window 는 550 까지 확장 허용."""
    n = len(es)
    for i in range(s0, min(s1, n)):
        j = i + sustain
        if j <= n and np.all(es[i:j] > thr):
            return i
    return None


def find_candidates(es, s0, s1, thr, sustain):
    n = len(es)
    cands = []
    for i in range(s0, min(s1, n)):
        j = i + sustain
        if j <= n and np.all(es[i:j] > thr):
            cands.append(i)
    return cands


def median(x):
    return float(np.median(x)) if len(x) else None


def mad(x):
    a = np.asarray(x, float)
    if a.size == 0:
        return None
    return float(np.median(np.abs(a - np.median(a))))


# ─── split 재현 (split_safesignal_within_subject 동일 로직, filename 단위) ────
def reproduce_within_subject_split():
    d = np.load(SPLIT_CACHE, allow_pickle=True)
    filenames = [str(x) for x in d["filename"].tolist()]
    subj = d["subject"].tolist()
    y = d["y"].tolist()
    session_key = {}
    for i, fn in enumerate(filenames):
        if any(m in Path(fn).stem.upper() for m in SIDE_FALL_MARKERS):
            continue
        session_key.setdefault(fn, (int(subj[i]), int(y[i])))
    strata = {}
    for fn, key in session_key.items():
        strata.setdefault(key, []).append(fn)
    rng = np.random.default_rng(SEED)
    assign = {}
    for key in sorted(strata):
        sessions = sorted(strata[key])
        rng.shuffle(sessions)
        n = len(sessions)
        n_test = max(1, int(round(n * TEST_RATIO))) if n >= 2 else 0
        n_test = min(n_test, n - 1) if n >= 1 else 0
        test = sessions[:n_test]
        pool = sessions[n_test:]
        m = len(pool)
        n_val = max(1, int(round(m * VAL_RATIO))) if m >= 2 else 0
        n_val = min(n_val, m - 1) if m >= 1 else 0
        val = pool[:n_val]
        train = pool[n_val:]
        for fn in train:
            assign[fn] = "train"
        for fn in val:
            assign[fn] = "val"
        for fn in test:
            assign[fn] = "test"
    return assign  # filename(.csv) -> 'train'|'val'|'test'


# ─── 세션 분석 ───────────────────────────────────────────────────────────────
def subtype_of(meta):
    return meta.activity.upper()  # e.g. FALL_WALK_B


def analyze_session(path, split_assign):
    meta = parse_safesignal_filename(path)
    fname = path.name
    subtype = subtype_of(meta)
    split_assignment = split_assign.get(fname, "out_of_scope")
    decision_scope = "metadata_only" if split_assignment == "out_of_scope" else SPLIT_SCOPE
    base = dict(
        manifest_id=MANIFEST_ID, manifest_version=MANIFEST_VERSION, filename=fname,
        env=meta.environment, subject=meta.subject, activity=meta.activity, subtype=subtype, trial=meta.trial,
        split_id=SPLIT_ID, split_assignment=split_assignment, split_source=SPLIT_SOURCE,
        split_unit=SPLIT_UNIT, decision_scope=decision_scope, test_sealed=True,
        thresholds_decided_from="train+val",
    )

    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    n = int(res.resampled_count)

    # 공통 review_priority (weak subtype)
    weak = subtype in WEAK_SUBTYPES

    if n < ORIGINAL_LEN:  # 검출 불가 — excluded/hard
        row = {**base, "resampled_count": n, "excluded": True, "exclude_reason": "resampled_count<550",
               "search_mode": "", "param_set_id": "",
               "needs_review": True, "needs_review_hard_reasons": "resampled_count<550",
               "needs_review_soft_reasons": "", "review_priority": "high" if (weak or True) else "normal",
               "review_priority_reasons": "hard_resampled_count_lt_550" + (";weak_subtype" if weak else "")}
        summ = {"filename": fname, "split_assignment": split_assignment, "subtype": subtype,
                "n_valid_param_rows": 0, "nominal_valid_count": 0, "broad_valid_count": 0,
                "nominal_rise_median": None, "broad_rise_median": None, "nominal_broad_delta_median": None,
                "param_sensitivity_range_overall": None, "any_hard_review": True,
                "hard_reason_union": "resampled_count<550", "review_priority": "high",
                "review_priority_reasons": "hard_resampled_count_lt_550" + (";weak_subtype" if weak else "")}
        return [row], summ

    A = res.amplitude[:ORIGINAL_LEN]
    sparse = rpca_sparse(A, max_iter=DEFAULT_MAX_ITER, tol=None)
    es = smooth_centered(frame_energy(sparse), SMOOTH)  # (550,)

    b0, b1 = BASELINE
    bvals = es[b0:b1]
    baseline_median = float(np.median(bvals))
    baseline_mad = float(np.median(np.abs(bvals - baseline_median)))
    baseline_iqr = float(np.percentile(bvals, 75) - np.percentile(bvals, 25))
    baseline_p75 = float(np.percentile(bvals, 75))
    baseline_p90 = float(np.percentile(bvals, 90))
    baseline_noise_ratio = baseline_p90 / (baseline_median + EPS)
    baseline_mad_ratio = baseline_mad / (baseline_median + EPS)

    SEARCHES = {"nominal": SEARCH_NOMINAL, "broad": SEARCH_BROAD}
    rows = []
    rise_by = {}  # (search_mode, k, sustain) -> rise_orig or None
    for mode, (s0, s1) in SEARCHES.items():
        for k in K_MADS:
            for sustain in SUSTAINS:
                thr = baseline_median + k * baseline_mad
                rise = find_rise(es, s0, s1, thr, sustain)
                cands = find_candidates(es, s0, s1, thr, sustain)
                hard, soft = [], []

                peak = None
                if rise is not None:
                    seg = es[rise:s1]
                    peak = int(rise + int(np.argmax(seg))) if len(seg) else None
                rise_by[(mode, k, sustain)] = rise

                rise_clean = to_clean(rise) if rise is not None else None
                peak_clean = to_clean(peak) if peak is not None else None

                # hard reasons
                if rise is None:
                    hard.append("rise_not_found")
                else:
                    if not (s0 <= rise < s1):
                        hard.append("rise_outside_search")
                    if in_beep(rise):
                        hard.append("rise_in_beep_region")
                    if rise < NOMINAL_ONSET_ORIG:
                        hard.append("rise_too_early_original_lt_200")
                    if rise_clean is None:
                        hard.append("frame_mapping_failed")
                if peak is None:
                    if rise is not None:
                        hard.append("peak_not_found")
                else:
                    if not (FALL_ORIG[0] <= peak < FALL_ORIG[1]):
                        hard.append("peak_outside_expected_fall_original_200_400")
                    if in_beep(peak):
                        hard.append("peak_in_beep_region")
                    if peak_clean is None:
                        hard.append("frame_mapping_failed")

                # top-k
                if cands:
                    cand_sorted = sorted(cands, key=lambda i: es[i], reverse=True)[:TOPK]
                    topk_spread = int(max(cand_sorted) - min(cand_sorted))
                    topk_multimodal = bool(topk_spread >= TOPK_MULTIMODAL_SPREAD and len(cands) >= 2)
                    if topk_multimodal and topk_spread >= TOPK_NO_CONSENSUS_SPREAD:
                        hard.append("topk_no_consensus_hard")
                else:
                    topk_spread, topk_multimodal = None, False

                # confidence components
                if rise is not None:
                    rise_strength = float(es[rise] / (baseline_median + EPS))
                    lo, hi = rise - SLOPE_WINDOW, rise + SLOPE_WINDOW
                    edge = lo < 0 or hi > len(es)
                    pre = es[max(0, lo):rise]
                    post = es[rise:min(len(es), hi)]
                    rise_slope = (float(np.median(post)) - float(np.median(pre))) / (SLOPE_WINDOW * (baseline_median + EPS)) \
                        if len(pre) and len(post) else None
                else:
                    rise_strength, rise_slope, edge = None, None, False
                peak_distance = abs(peak_clean - NOMINAL_ONSET_CLEAN) if peak_clean is not None else None

                rise_outside_nominal = bool(rise is not None and not (SEARCH_NOMINAL[0] <= rise < SEARCH_NOMINAL[1]))
                peak_outside_nominal = bool(peak is not None and not (SEARCH_NOMINAL[0] <= peak < SEARCH_NOMINAL[1]))

                dedup_hard = sorted(set(hard))
                rows.append({
                    **base, "resampled_count": n, "excluded": False, "exclude_reason": "",
                    "search_mode": mode, "search_start_original": s0, "search_end_original": s1,
                    "smooth_frames": SMOOTH, "k_mad": k, "sustain_frames": sustain,
                    "param_set_id": f"k{k}_s{sustain}",
                    "onset_frame_original": rise, "onset_frame_clean": rise_clean,
                    "rise_frame_original": rise, "rise_frame_clean": rise_clean,
                    "peak_frame_original": peak, "peak_frame_clean": peak_clean,
                    "baseline_median": baseline_median, "baseline_mad": baseline_mad, "baseline_iqr": baseline_iqr,
                    "baseline_p75": baseline_p75, "baseline_p90": baseline_p90,
                    "baseline_noise_ratio": baseline_noise_ratio, "baseline_mad_ratio": baseline_mad_ratio,
                    "rise_strength": rise_strength, "rise_slope": rise_slope, "slope_window": SLOPE_WINDOW,
                    "slope_edge_clipped": bool(edge), "peak_distance": peak_distance,
                    "topk_spread": topk_spread, "topk_multimodal": topk_multimodal,
                    "param_sensitivity_range": None, "param_sensitivity_mad": None, "confidence_ref": None,
                    "rise_outside_nominal": rise_outside_nominal, "peak_outside_nominal": peak_outside_nominal,
                    "nominal_broad_delta": None,
                    "needs_review": bool(dedup_hard), "needs_review_hard_reasons": ";".join(dedup_hard),
                    "needs_review_soft_reasons": "",  # soft threshold 미확정 — raw 값은 컬럼 참조
                    "review_priority": "", "review_priority_reasons": "",
                })

    # ── 세션 단위 후처리: param_sensitivity, nominal_broad_delta ──────────────
    def valid_rises(mode):
        return [rise_by[(mode, k, s)] for k in K_MADS for s in SUSTAINS if rise_by[(mode, k, s)] is not None]

    sens = {}
    for mode in SEARCHES:
        vr = valid_rises(mode)
        sens[mode] = (float(max(vr) - min(vr)) if len(vr) >= 2 else (0.0 if vr else None),
                      mad(vr) if vr else None, len(vr))
    all_vr = [rise_by[k] for k in rise_by if rise_by[k] is not None]
    sens_overall_range = float(max(all_vr) - min(all_vr)) if len(all_vr) >= 2 else (0.0 if all_vr else None)

    for r in rows:
        mode = r["search_mode"]
        if mode in sens:
            r["param_sensitivity_range"], r["param_sensitivity_mad"], _ = sens[mode]
        # nominal_broad_delta = rise_broad - rise_nominal (같은 param_set)
        if mode:
            k, sustain = r["k_mad"], r["sustain_frames"]
            rn, rb = rise_by.get(("nominal", k, sustain)), rise_by.get(("broad", k, sustain))
            r["nominal_broad_delta"] = (rb - rn) if (rn is not None and rb is not None) else None

    # ── review_priority (세션 단위) ──────────────────────────────────────────
    any_hard = any(r["needs_review"] for r in rows)
    hard_union = sorted({h for r in rows for h in (r["needs_review_hard_reasons"].split(";") if r["needs_review_hard_reasons"] else [])})
    pr_reasons = []
    if weak:
        pr_reasons.append("weak_subtype:" + subtype)
    if any_hard:
        pr_reasons.append("has_hard_review")
    priority = "high" if (weak or any_hard) else "normal"
    for r in rows:
        r["review_priority"] = priority
        r["review_priority_reasons"] = ";".join(pr_reasons)

    summ = {
        "filename": fname, "split_assignment": split_assignment, "subtype": subtype,
        "n_valid_param_rows": len(all_vr),
        "nominal_valid_count": sens["nominal"][2], "broad_valid_count": sens["broad"][2],
        "nominal_rise_median": median(valid_rises("nominal")),
        "broad_rise_median": median(valid_rises("broad")),
        "nominal_broad_delta_median": median([r["nominal_broad_delta"] for r in rows
                                              if r["nominal_broad_delta"] is not None]),
        "param_sensitivity_range_overall": sens_overall_range,
        "any_hard_review": any_hard, "hard_reason_union": ";".join(hard_union),
        "review_priority": priority, "review_priority_reasons": ";".join(pr_reasons),
        "_es": es, "_baseline_median": baseline_median, "_baseline_mad": baseline_mad,
    }
    return rows, summ


# ─── confidence_ref (전 valid row min-max 정규화, 정렬/우선순위용만) ──────────
def fill_confidence_ref(rows):
    def col(key):
        return [r[key] for r in rows if r.get(key) is not None and r["peak_distance"] is not None
                and r["rise_strength"] is not None]
    def mm(vals):
        a = np.asarray(vals, float)
        return (float(a.min()), float(a.max())) if a.size else (0.0, 1.0)
    rng_rs = mm(col("rise_strength")); rng_sl = mm([r["rise_slope"] for r in rows if r["rise_slope"] is not None])
    rng_pd = mm(col("peak_distance")); rng_ts = mm([r["topk_spread"] for r in rows if r["topk_spread"] is not None])
    rng_bn = mm([r["baseline_noise_ratio"] for r in rows])

    def norm(v, rng, invert=False):
        lo, hi = rng
        if hi <= lo or v is None:
            return None
        z = (v - lo) / (hi - lo)
        return 1 - z if invert else z

    for r in rows:
        if r["rise_strength"] is None or r["peak_distance"] is None:
            r["confidence_ref"] = None
            continue
        parts = [norm(r["rise_strength"], rng_rs), norm(r["rise_slope"], rng_sl),
                 norm(r["peak_distance"], rng_pd, invert=True),
                 norm(r["topk_spread"], rng_ts, invert=True) if r["topk_spread"] is not None else None,
                 norm(r["baseline_noise_ratio"], rng_bn, invert=True)]
        parts = [p for p in parts if p is not None]
        r["confidence_ref"] = float(np.mean(parts)) if parts else None


# ─── CSV ─────────────────────────────────────────────────────────────────────
LONG_COLS = ["manifest_id", "manifest_version", "filename", "env", "subject", "activity", "subtype", "trial",
             "split_id", "split_assignment", "split_source", "split_unit", "decision_scope", "test_sealed",
             "thresholds_decided_from", "resampled_count", "excluded", "exclude_reason",
             "search_mode", "search_start_original", "search_end_original", "smooth_frames", "k_mad",
             "sustain_frames", "param_set_id", "onset_frame_original", "onset_frame_clean", "rise_frame_original",
             "rise_frame_clean", "peak_frame_original", "peak_frame_clean", "baseline_median", "baseline_mad",
             "baseline_iqr", "baseline_p75", "baseline_p90", "baseline_noise_ratio", "baseline_mad_ratio",
             "rise_strength", "rise_slope", "slope_window", "slope_edge_clipped", "peak_distance", "topk_spread",
             "topk_multimodal", "param_sensitivity_range", "param_sensitivity_mad", "confidence_ref",
             "rise_outside_nominal", "peak_outside_nominal", "nominal_broad_delta", "needs_review",
             "needs_review_hard_reasons", "needs_review_soft_reasons", "review_priority", "review_priority_reasons"]
SUMM_COLS = ["filename", "split_assignment", "subtype", "n_valid_param_rows", "nominal_valid_count",
             "broad_valid_count", "nominal_rise_median", "broad_rise_median", "nominal_broad_delta_median",
             "param_sensitivity_range_overall", "any_hard_review", "hard_reason_union", "review_priority",
             "review_priority_reasons"]


def _fmt(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 5) if np.isfinite(v) else ""
    return "" if v is None else v


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in cols])


# ─── 집계/요약 ───────────────────────────────────────────────────────────────
def dist(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="(smoke) 세션 수 제한")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT / "plots"
    if HAVE_MPL:
        plots_dir.mkdir(parents=True, exist_ok=True)

    split_assign = reproduce_within_subject_split()
    n_tr = sum(1 for v in split_assign.values() if v == "train")
    n_va = sum(1 for v in split_assign.values() if v == "val")
    n_te = sum(1 for v in split_assign.values() if v == "test")
    print(f"[split] cache={SPLIT_CACHE.name} files={len(split_assign)} train={n_tr} val={n_va} test={n_te}")

    files = sorted(CLEANED.rglob("*FALL*_T*.csv"))
    files = [f for f in files if not any(m in f.stem.upper() for m in SIDE_FALL_MARKERS)]
    if args.limit:
        files = files[:args.limit]
    print(f"[scan] fall 세션 {len(files)}개 (side-fall 제외)")

    long_rows, summaries = [], []
    for i, f in enumerate(files):
        rows, summ = analyze_session(f, split_assign)
        long_rows.extend(rows)
        summaries.append(summ)
        if (i + 1) % 30 == 0 or i + 1 == len(files):
            print(f"  [{i+1}/{len(files)}] {f.name} split={summ['split_assignment']} "
                  f"hard={summ['any_hard_review']} prio={summ['review_priority']}")

    fill_confidence_ref([r for r in long_rows if not r["excluded"]])

    write_csv(OUT / "onset_probe_manifest_long.csv", long_rows, LONG_COLS)
    write_csv(OUT / "onset_probe_session_summary.csv", summaries, SUMM_COLS)

    # ── summary json/md ──────────────────────────────────────────────────────
    inc = [s for s in summaries if s["n_valid_param_rows"] >= 0 and s["split_assignment"] != "out_of_scope"]
    by_split = {sp: [s for s in summaries if s["split_assignment"] == sp] for sp in ("train", "val", "test", "out_of_scope")}
    detect_rows = [r for r in long_rows if not r["excluded"]]
    excluded_sessions = [s for s in summaries if s["hard_reason_union"] == "resampled_count<550" and s["n_valid_param_rows"] == 0]

    # hard reason 건수 (param row 기준)
    hard_counts = {}
    for r in detect_rows:
        for h in (r["needs_review_hard_reasons"].split(";") if r["needs_review_hard_reasons"] else []):
            hard_counts[h] = hard_counts.get(h, 0) + 1

    summary = {
        "meta": {"manifest_id": MANIFEST_ID, "split_id": SPLIT_ID, "split_cache": SPLIT_CACHE.name,
                 "baseline": list(BASELINE), "search_nominal": list(SEARCH_NOMINAL),
                 "search_broad": list(SEARCH_BROAD), "grid": {"smooth": SMOOTH, "k_mad": K_MADS, "sustain": SUSTAINS},
                 "note": "probe — search/k/M/soft threshold 미확정. soft reason은 train/val 분포 후속 결정."},
        "session_counts": {
            "total": len(summaries),
            "by_split": {k: len(v) for k, v in by_split.items()},
            "excluded_lt550": len(excluded_sessions),
            "by_subtype": {st: sum(1 for s in summaries if s["subtype"] == st)
                           for st in sorted({s["subtype"] for s in summaries})},
        },
        "onset_distribution": {
            "rise_frame_clean": dist([r["rise_frame_clean"] for r in detect_rows]),
            "peak_frame_clean": dist([r["peak_frame_clean"] for r in detect_rows]),
            "by_subtype_rise_clean": {st: dist([r["rise_frame_clean"] for r in detect_rows if r["subtype"] == st])
                                      for st in sorted({r["subtype"] for r in detect_rows})},
        },
        "search_compare": {
            "rise_outside_nominal_rate": float(np.mean([1.0 if r["rise_outside_nominal"] else 0.0 for r in detect_rows])) if detect_rows else None,
            "nominal_broad_delta": dist([r["nominal_broad_delta"] for r in detect_rows]),
            "broad_only_session_rate": float(np.mean([
                1.0 if (s["nominal_valid_count"] == 0 and s["broad_valid_count"] > 0) else 0.0 for s in summaries])) if summaries else None,
        },
        "param_stability": {
            "param_sensitivity_range_overall": dist([s["param_sensitivity_range_overall"] for s in summaries]),
            "by_subtype": {st: dist([s["param_sensitivity_range_overall"] for s in summaries if s["subtype"] == st])
                           for st in sorted({s["subtype"] for s in summaries})},
        },
        "needs_review": {
            "hard_reason_param_row_counts": hard_counts,
            "sessions_any_hard": int(sum(1 for s in summaries if s["any_hard_review"])),
            "review_priority_high_sessions": int(sum(1 for s in summaries if s["review_priority"] == "high")),
            "high_by_subtype": {st: int(sum(1 for s in summaries if s["subtype"] == st and s["review_priority"] == "high"))
                                for st in sorted({s["subtype"] for s in summaries})},
        },
        "baseline_quality": {
            "noise_ratio": dist([r["baseline_noise_ratio"] for r in detect_rows]),
            "mad_ratio": dist([r["baseline_mad_ratio"] for r in detect_rows]),
        },
        "soft_components_raw_dist": {
            "rise_strength": dist([r["rise_strength"] for r in detect_rows]),
            "rise_slope": dist([r["rise_slope"] for r in detect_rows]),
            "topk_spread": dist([r["topk_spread"] for r in detect_rows]),
            "confidence_ref": dist([r["confidence_ref"] for r in detect_rows]),
        },
    }
    (OUT / "onset_probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# 게이트 2 onset detector probe 요약\n",
         f"split: {SPLIT_ID} (cache {SPLIT_CACHE.name})",
         f"세션: total {len(summaries)} | " + " ".join(f"{k}={len(v)}" for k, v in by_split.items()) +
         f" | excluded<550={len(excluded_sessions)}",
         f"subtype별: {summary['session_counts']['by_subtype']}\n",
         "## onset 분포 (clean frame, nominal onset=100)",
         f"- rise_frame_clean: {summary['onset_distribution']['rise_frame_clean']}",
         f"- peak_frame_clean: {summary['onset_distribution']['peak_frame_clean']}",
         "## search 비교",
         f"- rise_outside_nominal_rate: {summary['search_compare']['rise_outside_nominal_rate']}",
         f"- broad_only_session_rate: {summary['search_compare']['broad_only_session_rate']}",
         f"- nominal_broad_delta: {summary['search_compare']['nominal_broad_delta']}",
         "## param 안정성",
         f"- param_sensitivity_range_overall: {summary['param_stability']['param_sensitivity_range_overall']}",
         "## needs_review",
         f"- hard reason 건수(param row): {hard_counts}",
         f"- any_hard 세션: {summary['needs_review']['sessions_any_hard']} / priority high 세션: {summary['needs_review']['review_priority_high_sessions']}",
         f"- high_by_subtype: {summary['needs_review']['high_by_subtype']}",
         "## baseline 품질",
         f"- noise_ratio: {summary['baseline_quality']['noise_ratio']}",
         f"- mad_ratio: {summary['baseline_quality']['mad_ratio']}",
         "## soft 구성요소 raw 분포 (threshold 미확정)",
         f"- rise_strength: {summary['soft_components_raw_dist']['rise_strength']}",
         f"- rise_slope: {summary['soft_components_raw_dist']['rise_slope']}",
         f"- topk_spread: {summary['soft_components_raw_dist']['topk_spread']}",
         f"- confidence_ref: {summary['soft_components_raw_dist']['confidence_ref']}"]
    (OUT / "onset_probe_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── plots ─────────────────────────────────────────────────────────────────
    if HAVE_MPL:
        rc = [r["rise_frame_clean"] for r in detect_rows if r["rise_frame_clean"] is not None]
        pc = [r["peak_frame_clean"] for r in detect_rows if r["peak_frame_clean"] is not None]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        if rc:
            ax[0].hist(rc, bins=30, color="C0"); ax[0].axvline(100, color="C3", ls="--", lw=1)
        ax[0].set_title("rise_frame_clean (onset=100)", fontsize=9); ax[0].set_xlabel("clean frame")
        if pc:
            ax[1].hist(pc, bins=30, color="C1"); ax[1].axvline(100, color="C3", ls="--", lw=1)
        ax[1].set_title("peak_frame_clean", fontsize=9); ax[1].set_xlabel("clean frame")
        fig.tight_layout(); fig.savefig(plots_dir / "onset_clean_hist.png", dpi=120); plt.close(fig)

    print("\n=== 산출 완료 ===")
    for f in ["onset_probe_manifest_long.csv", "onset_probe_session_summary.csv",
              "onset_probe_summary.json", "onset_probe_summary.md"]:
        print(f"  {OUT / f}")
    print("\n----- onset_probe_summary.md -----")
    print((OUT / "onset_probe_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
