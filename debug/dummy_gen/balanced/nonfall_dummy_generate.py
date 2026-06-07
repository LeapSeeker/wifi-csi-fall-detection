"""balanced 증강용 신규 non-fall 더미 생성기 (read-only 입력, 신규 파생물만).

기존 fall-only 더미는 fall:non-fall 비를 키워 class-ratio artifact 를 만든다. 이를 통제하기
위해 train non-fall 윈도우 기반 더미를 추가 생성한다(C/D 공유 단일 set).

핵심(동결 cache 가 raw 를 안 들고 있음):
- item4 cache 의 non-fall 행 X 는 raw window 가 아니라 이미 z-SDP (1,28,20)다.
- 따라서 원본 CSV 에서 raw window (300,104) 를 *재생성*하여 cache 행과 1:1 정합을 검증한 뒤
  (count 일치 + window_to_model_input allclose), raw 를 perturb → window_to_model_input 으로
  더미 z-SDP 를 만든다. z-SDP 직접 변형 금지(스펙).

origin 집합: item4_cache_fixed.npz 의 split_assignment=="train" AND y!=0 (fixed/onset_primary
non-fall 동일집합 → 단일 set, C/D 공유). non-fall split source 는 item4 fixed cache 의
canonical seed42 split(=manifest_v2 아님).

profile v1: speed{normal,fast} × body{small,medium,large} = 6, deterministic round-robin.
생성량: train non-fall raw window 당 최대 1개. seed = (BASE_SEED, origin_cache_row_idx) → 재현.

품질 gate: 원본 train non-fall 만으로 class별 robust center/MAD → robust distance(per-dim
robust z 의 median). default threshold = class별 robust distance p97.5 (p95 report-only).
use 조건: finite & robust_distance<=p97.5 & raw amplitude mean ratio∈[0.5,2.0]. onset 지표 미사용.
class별 used < target*0.5 → low_diversity_contribution=true (eval 결론부 자동 반영).

산출:
- nonfall_dummies.npz : use 더미만 (X(N,1,28,20), y, filename, subject, env, activity, trial,
  is_augmented, origin_cache_row_idx, origin_window_start)
- nonfall_lineage.csv : 전 후보(use/reject) + 사유 + 파라미터
- nonfall_quality_report.json|md

제약(read-only): 원본 CSV·기존 cache/manifest·동결 파일 무수정. window_to_model_input/perturb
함수는 import/재사용만. 더미 filename 에 반드시 ``__aug`` 포함.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "debug/modeling"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from model.preprocessing.pipeline import preprocess_safesignal_file, window_to_model_input  # noqa: E402
from debug.dummy_gen.dummy_clean400_lib import (  # noqa: E402
    BODY_PARAMS, SPEED_PARAMS, NOISE_SNR_DB,
    _amplitude_scale, _time_warp, _subcarrier, _add_noise,
)

# ── 경로/상수 ─────────────────────────────────────────────────────────────────
ITEM4 = ROOT / "debug/modeling/diag_out/onset_detector/item4"
FIXED_CACHE = ITEM4 / "item4_cache_fixed.npz"
CLEANED = ROOT / "data/cleaned"
OUTDIR = ROOT / "debug/dummy_gen/balanced"
OUT_NPZ = OUTDIR / "nonfall_dummies.npz"
OUT_LIN = OUTDIR / "nonfall_lineage.csv"
OUT_QJSON = OUTDIR / "nonfall_quality_report.json"
OUT_QMD = OUTDIR / "nonfall_quality_report.md"

WINDOW = 300
N_SC = 104
BASE_SEED = 20260608
# profile round-robin 순서 (deterministic): body × speed
PROFILES = [(b, s) for b in ("small", "medium", "large") for s in ("normal", "fast")]
AMP_RATIO_LO, AMP_RATIO_HI = 0.5, 2.0
# preprocess_safesignal_file 파라미터 (orig cache 빌드와 동일해야 count/feature 정합)
PRE_KW = dict(rx="both", target_hz=100.0, max_gap_ms=100.0, window_size=WINDOW,
              stride=None, tail_window=True, pad_short=False)
LINEAGE_COLS = [
    "aug_filename", "origin_cache", "origin_cache_row_idx", "origin_filename",
    "origin_group_key", "origin_window_start", "origin_window_idx", "origin_split",
    "origin_subject", "origin_env", "origin_trial", "origin_activity", "origin_y",
    "transform_profile", "transform_params", "warp_factor", "scale_factor", "noise_snr",
    "amplitude_ratio", "quality_distance", "quality_threshold", "aug_status",
    "reject_or_pending_reason", "is_augmented",
]


def window_starts(n_packets, window_size=WINDOW, stride=WINDOW, tail_window=True):
    """sliding_windows 와 동일한 윈도우 시작 frame 목록(원본 좌표). PRE_KW(stride300/tail) 대응."""
    if n_packets < window_size:
        return [0]
    n_windows = 1 + (n_packets - window_size) // stride
    starts = [int(i * stride) for i in range(n_windows)]
    last_end = starts[-1] + window_size
    if tail_window and last_end < n_packets:
        starts.append(int(n_packets - window_size))   # tail = amplitude[-window_size:]
    return starts


def preflight():
    miss = [p for p in (FIXED_CACHE,) if not p.exists()]
    if not CLEANED.is_dir():
        miss.append(CLEANED)
    if miss:
        print("[FAIL-FAST] 필수 입력 없음:")
        for p in miss:
            print(f"  - {p}")
        raise SystemExit(2)


# ── perturb worker (모듈 레벨, picklable) ─────────────────────────────────────
def _perturb_to_feat(args):
    """raw window (300,104) → profile perturb → (feat(1,28,20), params, amp_ratio)."""
    win, body, speed, seed = args
    rng = np.random.default_rng(seed)
    rx1 = win[:, :52].astype(np.float64).copy()
    rx2 = win[:, 52:].astype(np.float64).copy()
    blo, bhi = BODY_PARAMS[body]
    slo, shi = SPEED_PARAMS[speed]
    rx1, scale = _amplitude_scale(rx1, rng, blo, bhi)
    rx2, _ = _amplitude_scale(rx2, rng, blo, bhi, scale=scale)
    rx1, factor, off = _time_warp(rx1, rng, slo, shi)
    rx2, _, _ = _time_warp(rx2, rng, slo, shi, factor=factor)
    rx1 = _subcarrier(rx1, rng)
    rx2 = _subcarrier(rx2, rng)
    rx1, snr = _add_noise(rx1, rng, *NOISE_SNR_DB)
    rx2, _ = _add_noise(rx2, rng, *NOISE_SNR_DB)
    aug = np.concatenate([rx1, rx2], axis=1)
    feat = window_to_model_input(aug).astype(np.float32)
    amp_ratio = float(aug.mean() / (win.mean() + 1e-9))
    params = {"scale_factor": round(float(scale), 5), "warp_factor": round(float(factor), 5),
              "crop_offset": int(off), "noise_snr": round(float(snr), 2)}
    return feat, params, amp_ratio


def _sanity_feat(win):
    """재생성 raw window → z-SDP (cache 행과 allclose 검증용)."""
    return window_to_model_input(win[:, :].astype(np.float64)).astype(np.float32)


def robust_center_mad(feats):
    """feats (n,1,28,20) → (center(560,), mad(560,))."""
    F = feats.reshape(len(feats), -1).astype(np.float64)
    center = np.median(F, axis=0)
    mad = np.median(np.abs(F - center), axis=0)
    mad[mad < 1e-9] = 1e-9
    return center, mad


def robust_distance(feat, center, mad):
    f = feat.reshape(-1).astype(np.float64)
    return float(np.median(np.abs(f - center) / mad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0, help="0=auto(min16,cpu-2)")
    ap.add_argument("--limit-files", type=int, default=None, help="(smoke) origin 파일 수 제한")
    ap.add_argument("--strict-allclose", action="store_true",
                    help="재생성 raw→z-SDP 가 cache 행과 불일치 시 fail-fast")
    ap.add_argument("--allclose-atol", type=float, default=1e-4)
    ap.add_argument("--allclose-rtol", type=float, default=1e-4)
    args = ap.parse_args()
    preflight()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    nw = args.workers or min(16, max(1, mp.cpu_count() - 2))

    d = np.load(FIXED_CACHE, allow_pickle=True)
    classes = list(d["classes"])
    sp = np.array([str(x) for x in d["split_assignment"]])
    y = d["y"].astype(int)
    fn = np.array([str(x) for x in d["filename"]])
    subj = d["subject"].astype(int)
    env = d["env"].astype(int)
    act = np.array([str(x) for x in d["activity"]])
    trial = d["trial"].astype(int)
    X = d["X"].astype(np.float32)

    train_nf = np.where((sp == "train") & (y != 0))[0]
    target_count = Counter(int(y[i]) for i in train_nf)
    print(f"[origin] train non-fall {len(train_nf)} | per-class "
          f"{ {classes[c]: target_count[c] for c in sorted(target_count)} }")

    # 원본 train non-fall feature 로 class별 robust center/MAD/threshold
    centers, mads, thr975, thr95 = {}, {}, {}, {}
    for c in sorted(target_count):
        rows = train_nf[y[train_nf] == c]
        feats = X[rows]
        ctr, mad = robust_center_mad(feats)
        dists = np.array([robust_distance(f, ctr, mad) for f in feats])
        centers[c], mads[c] = ctr, mad
        thr975[c] = float(np.percentile(dists, 97.5))
        thr95[c] = float(np.percentile(dists, 95.0))
    print(f"[quality] class별 robust distance p97.5={ {classes[c]: round(thr975[c],3) for c in thr975} }")

    # filename별 cache 행 그룹(순서 보존 = window 순서)
    by_file = OrderedDict()
    for i in train_nf:
        by_file.setdefault(fn[i], []).append(int(i))
    files = list(by_file.keys())
    if args.limit_files:
        files = files[:args.limit_files]
        print(f"[smoke] origin 파일 {len(files)} 개로 제한")

    # ── raw window 재생성 + cache 정합 검증 ───────────────────────────────────
    regen = {}        # filename -> windows (n,300,104)
    starts_map = {}   # filename -> [window start frame(원본 좌표)]
    csv_missing, count_mismatch = [], []
    for f in files:
        p = next(CLEANED.rglob(f), None)
        if p is None:
            csv_missing.append(f)
            continue
        res = preprocess_safesignal_file(p, **PRE_KW)
        wins = np.asarray(res.windows)
        rows = by_file[f]
        if len(wins) != len(rows):
            count_mismatch.append((f, len(wins), len(rows)))
            continue
        n_packets = int(res.resample.amplitude.shape[0])
        starts = window_starts(n_packets)
        if len(starts) != len(wins):   # sliding_windows 규칙과 어긋나면 frame start 신뢰 불가
            count_mismatch.append((f, len(wins), f"starts{len(starts)}"))
            continue
        regen[f] = wins
        starts_map[f] = starts
    if csv_missing:
        print(f"[FAIL-FAST] CSV 미발견 {len(csv_missing)}: {csv_missing[:5]}")
        raise SystemExit(3)
    if count_mismatch:
        print(f"[FAIL-FAST] window count != cache row count {len(count_mismatch)}:")
        for f, a, b in count_mismatch[:8]:
            print(f"  {f}: regen {a} vs cache {b}")
        raise SystemExit(3)
    print(f"[regen] {len(regen)} 파일 raw window 재생성, cache count 일치 OK")

    # allclose sanity (재생성 raw→z-SDP vs cache 행) — 병렬
    sanity_tasks, sanity_key = [], []
    for f in files:
        for k, ri in enumerate(by_file[f]):
            sanity_tasks.append(regen[f][k]); sanity_key.append((f, k, ri))
    print(f"[sanity] {len(sanity_tasks)} window allclose 검증 (atol={args.allclose_atol})...", flush=True)
    max_abs_diff = 0.0
    n_mismatch = 0
    sfeats = [None] * len(sanity_tasks)
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futs = {pool.submit(_sanity_feat, t): i for i, t in enumerate(sanity_tasks)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]; sfeats[i] = fut.result(); done += 1
            if done % 200 == 0 or done == len(sanity_tasks):
                print(f"    sanity {done}/{len(sanity_tasks)}", flush=True)
    for i, (f, k, ri) in enumerate(sanity_key):
        diff = float(np.max(np.abs(sfeats[i] - X[ri])))
        max_abs_diff = max(max_abs_diff, diff)
        if not np.allclose(sfeats[i], X[ri], atol=args.allclose_atol, rtol=args.allclose_rtol):
            n_mismatch += 1
    print(f"[sanity] max_abs_diff={max_abs_diff:.3e} | mismatch {n_mismatch}/{len(sanity_key)}")
    if n_mismatch and args.strict_allclose:
        print("[FAIL-FAST] --strict-allclose: 재생성↔cache 불일치"); raise SystemExit(4)
    if n_mismatch:
        print("[warn] 재생성↔cache 불일치 존재 — report warning 으로 진행(스펙 기본)")

    # ── 후보 더미 생성 (window당 1개, profile round-robin) ─────────────────────
    cand = []   # dict per candidate
    pidx = 0
    for f in files:
        for k, ri in enumerate(by_file[f]):
            body, speed = PROFILES[pidx % len(PROFILES)]
            pidx += 1
            cand.append({"origin_filename": f, "origin_cache_row_idx": ri,
                         "origin_window_start": starts_map[f][k], "origin_window_idx": k,
                         "body": body, "speed": speed, "win": regen[f][k]})
    print(f"[candidate] {len(cand)} 후보 (window당 1개)")

    # perturb + feature (병렬)
    tasks = [(c["win"], c["body"], c["speed"], (BASE_SEED, c["origin_cache_row_idx"])) for c in cand]
    feats = [None] * len(tasks); params = [None] * len(tasks); amps = [None] * len(tasks)
    print(f"[feature] perturb+window_to_model_input {len(tasks)} (workers={nw})...", flush=True)
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futs = {pool.submit(_perturb_to_feat, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]; feats[i], params[i], amps[i] = fut.result(); done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"    feat {done}/{len(tasks)}", flush=True)

    # ── 품질 gate + lineage ───────────────────────────────────────────────────
    cache_meta = {int(i): {"subject": int(subj[i]), "env": int(env[i]), "activity": str(act[i]),
                           "trial": int(trial[i]), "y": int(y[i]), "split": str(sp[i])}
                  for i in train_nf}
    lineage_rows = []
    use_idx = []
    used_per_class = Counter()
    for i, c in enumerate(cand):
        ri = c["origin_cache_row_idx"]; cm = cache_meta[ri]
        cls = cm["y"]
        feat = feats[i]
        finite = bool(np.all(np.isfinite(feat)))
        dist = robust_distance(feat, centers[cls], mads[cls]) if finite else float("inf")
        thr = thr975[cls]
        amp = amps[i]
        reasons = []
        if not finite:
            reasons.append("nonfinite")
        if dist > thr:
            reasons.append(f"robust_dist>{thr:.3f}({dist:.3f})")
        if not (AMP_RATIO_LO <= amp <= AMP_RATIO_HI):
            reasons.append(f"amp_ratio_oob({amp:.3f})")
        status = "use" if not reasons else "reject"
        stem = Path(c["origin_filename"]).stem
        aug_fn = f"{stem}__augnf_{c['body'][:2]}_{c['speed'][:2]}_w{c['origin_window_start']}.npz"
        if status == "use":
            use_idx.append(i); used_per_class[cls] += 1
        lineage_rows.append({
            "aug_filename": aug_fn, "origin_cache": "item4_cache_fixed.npz",
            "origin_cache_row_idx": ri, "origin_filename": c["origin_filename"],
            "origin_group_key": stem, "origin_window_start": c["origin_window_start"],
            "origin_window_idx": c["origin_window_idx"],
            "origin_split": cm["split"], "origin_subject": cm["subject"], "origin_env": cm["env"],
            "origin_trial": cm["trial"], "origin_activity": cm["activity"], "origin_y": cls,
            "transform_profile": f"{c['body']}_{c['speed']}",
            "transform_params": f"scale{params[i]['scale_factor']}_warp{params[i]['warp_factor']}"
                                f"_off{params[i]['crop_offset']}_snr{params[i]['noise_snr']}",
            "warp_factor": params[i]["warp_factor"], "scale_factor": params[i]["scale_factor"],
            "noise_snr": params[i]["noise_snr"], "amplitude_ratio": round(amp, 5),
            "quality_distance": round(dist, 5) if np.isfinite(dist) else "inf",
            "quality_threshold": round(thr, 5), "aug_status": status,
            "reject_or_pending_reason": ";".join(reasons), "is_augmented": True,
        })

    # ── npz (use 만) ──────────────────────────────────────────────────────────
    if use_idx:
        Xo = np.stack([feats[i] for i in use_idx]).astype(np.float32)
    else:
        Xo = np.empty((0, 1, 28, 20), np.float32)
    def col(key, cast):
        return np.asarray([cast(lineage_rows[i][key]) for i in use_idx])
    np.savez_compressed(
        OUT_NPZ,
        X=Xo,
        y=np.asarray([cache_meta[cand[i]["origin_cache_row_idx"]]["y"] for i in use_idx], np.int64),
        filename=np.asarray([lineage_rows[i]["aug_filename"] for i in use_idx], object),
        subject=np.asarray([cache_meta[cand[i]["origin_cache_row_idx"]]["subject"] for i in use_idx], np.int64),
        env=np.asarray([cache_meta[cand[i]["origin_cache_row_idx"]]["env"] for i in use_idx], np.int64),
        activity=np.asarray([cache_meta[cand[i]["origin_cache_row_idx"]]["activity"] for i in use_idx], object),
        trial=np.asarray([cache_meta[cand[i]["origin_cache_row_idx"]]["trial"] for i in use_idx], np.int64),
        is_augmented=np.ones(len(use_idx), bool),
        origin_cache_row_idx=np.asarray([cand[i]["origin_cache_row_idx"] for i in use_idx], np.int64),
        origin_window_start=np.asarray([cand[i]["origin_window_start"] for i in use_idx], np.int64),
        origin_window_idx=np.asarray([cand[i]["origin_window_idx"] for i in use_idx], np.int64),
    )

    # lineage csv
    with open(OUT_LIN, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=LINEAGE_COLS); w.writeheader()
        for r in lineage_rows:
            w.writerow(r)

    # ── 품질 리포트 + low_diversity ───────────────────────────────────────────
    low_div = {}
    per_class_report = {}
    for c in sorted(target_count):
        used = used_per_class[c]; tgt = target_count[c]
        ld = used < 0.5 * tgt
        low_div[classes[c]] = ld
        per_class_report[classes[c]] = {
            "target_count": tgt, "candidates": tgt if not args.limit_files else None,
            "used": used, "use_ratio": round(used / tgt, 4) if tgt else 0.0,
            "threshold_p975": round(thr975[c], 5), "threshold_p95": round(thr95[c], 5),
            "low_diversity_contribution": ld,
        }
    n_use = len(use_idx); n_rej = len(cand) - n_use
    report = {
        "base_seed": BASE_SEED, "profiles": [f"{b}_{s}" for b, s in PROFILES],
        "n_candidates": len(cand), "n_use": n_use, "n_reject": n_rej,
        "use_ratio_overall": round(n_use / len(cand), 4) if cand else 0.0,
        "sanity_max_abs_diff": max_abs_diff, "sanity_mismatch": n_mismatch,
        "amp_ratio_bounds": [AMP_RATIO_LO, AMP_RATIO_HI],
        "quality_metric": "robust distance = median over dims of |f-center|/MAD (per-class, orig train non-fall)",
        "threshold_policy": "class별 robust distance p97.5 (default), p95 report-only",
        "per_class": per_class_report,
        "low_diversity_contribution": low_div,
        "low_diversity_classes": [k for k, v in low_div.items() if v],
        "reject_reason_counts": dict(Counter(
            r["reject_or_pending_reason"] for r in lineage_rows if r["aug_status"] == "reject")),
        "limit_files": args.limit_files,
    }
    OUT_QJSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    L = ["# non-fall 더미 품질 리포트 (balanced)\n",
         f"후보 {len(cand)} | use {n_use} | reject {n_rej} | use율 {report['use_ratio_overall']}",
         f"sanity 재생성↔cache max_abs_diff={max_abs_diff:.3e}, mismatch {n_mismatch}",
         "\n## class별 use / target", "class | target | used | use율 | p97.5 | low_div",
         "---|---|---|---|---|---"]
    for c in sorted(target_count):
        r = per_class_report[classes[c]]
        L.append(f"{classes[c]} | {r['target_count']} | {r['used']} | {r['use_ratio']} | "
                 f"{r['threshold_p975']} | {'⚠TRUE' if r['low_diversity_contribution'] else 'false'}")
    if report["low_diversity_classes"]:
        L.append(f"\n⚠ low_diversity_contribution=true: {report['low_diversity_classes']} "
                 f"— 해당 class 다양성 기여 제한적. eval 결론부 자동 반영.")
    OUT_QMD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\n[생성] {OUT_NPZ} (use {n_use})")
    print(f"[생성] {OUT_LIN} ({len(lineage_rows)} 후보)")
    print(f"[생성] {OUT_QJSON}, {OUT_QMD}")
    print(f"  use/candidate {n_use}/{len(cand)} | low_div {report['low_diversity_classes']}")
    print(f"  used per class: { {classes[c]: used_per_class[c] for c in sorted(target_count)} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
