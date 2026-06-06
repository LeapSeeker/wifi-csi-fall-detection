"""더미 직접 생성 1차 — clean400 non-WALK train fall ×3 + onset 교차검증 + 3단계 분류.

origin: train non-WALK fall(auto/manual 확정 onset)만. ×3 = 9 profile(body×speed) 균등 round-robin.
clean400 위에서 증강(scale/time_warp/subcarrier/noise, 보수 SNR), 난수 파라미터 전부 기록.
expected(해석적) vs detected(clean400 detector) onset 교차검증 → 사용/pending/제외.

산출(gitignore): out/dummies_clean400.npz (aug400 + 메타), out/lineage.csv (전 더미 lineage).
학습 시작 금지. read-only(원본·manifest·동결파일 무수정).
"""
from __future__ import annotations
import argparse
import csv
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "debug/dummy_gen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import dummy_clean400_lib as L  # noqa: E402

V2 = ROOT / "debug/modeling/diag_out/onset_detector/finalization/manifest_v2_manual_augmented.csv"
CLEANED = ROOT / "data/cleaned"
WALK = {"FALL_WALK_F", "FALL_WALK_B"}
NOISE_HI = 1.30  # baseline_noise_ratio flag
BODIES = ["small", "medium", "large"]


def select_origins():
    rows = list(csv.DictReader(open(V2, encoding="utf-8-sig")))
    return [r for r in rows
            if r["split_assignment"] == "train" and r["subtype"] not in WALK
            and r["onset_status"] in ("auto_reviewed", "manual_corrected")
            and r["onset_frame_clean"] not in ("", "None")]


def _worker(task):
    (fn, onset_clean, subtype, subject, env, status, prof, aug_id, seed) = task
    out = {"aug_filename": f"{Path(fn).stem}__aug{aug_id}_{prof['body_size'][:2]}_{prof['fall_speed'][:2]}.npz",
           "origin_filename": fn, "origin_split": "train", "origin_subtype": subtype,
           "origin_subject": subject, "origin_env": env, "aug_id": aug_id,
           "transform_profile": f"{prof['body_size']}_{prof['fall_speed']}",
           "transform_params": "", "warp_factor": "", "scale_factor": "", "noise_snr": "",
           "crop_offset": "", "splice_boundaries_clean": "100|300",
           "expected_onset_clean": "", "detected_onset_clean": "", "onset_delta": "",
           "baseline_noise_ratio": "", "auto_augmented_onset_ok": False,
           "aug_status": "", "reject_or_pending_reason": ""}
    p = next(CLEANED.rglob(fn), None)
    if p is None:
        out["aug_status"] = "exclude"; out["reject_or_pending_reason"] = "origin_csv_not_found"
        return out, None
    c4 = L.build_clean400(p)
    if c4 is None:
        out["aug_status"] = "exclude"; out["reject_or_pending_reason"] = "data_invalid_resampled_lt_550"
        return out, None
    rng = np.random.default_rng(seed)
    aug, params = L.augment_clean400(c4, prof, rng)
    out["warp_factor"] = params["warp_factor"]; out["scale_factor"] = params["scale_factor"]
    out["noise_snr"] = params["noise_snr"]; out["crop_offset"] = params["crop_offset"]
    out["transform_params"] = f"scale{params['scale_factor']}_warp{params['warp_factor']}_off{params['crop_offset']}_snr{params['noise_snr']}"
    exp = L.expected_onset(onset_clean, params["warp_factor"], params["crop_offset"])
    det = L.detect_onset_clean(aug)
    out["baseline_noise_ratio"] = round(det["baseline_noise_ratio"], 4)
    out["expected_onset_clean"] = "" if exp is None else exp
    out["detected_onset_clean"] = "" if det["rise"] is None else det["rise"]
    is_warp = prof["fall_speed"] in ("fast", "slow")
    tol = 10 if is_warp else 5
    # 3단계 분류
    if exp is None:
        out["aug_status"] = "pending"; out["reject_or_pending_reason"] = "crop_out_of_bounds"
    elif det["rise"] is None:
        out["aug_status"] = "pending"; out["reject_or_pending_reason"] = "rise_not_found"
    elif det["baseline_noise_ratio"] > NOISE_HI:
        out["aug_status"] = "pending"; out["reject_or_pending_reason"] = "baseline_noise_high"
    else:
        delta = int(det["rise"]) - int(exp)
        out["onset_delta"] = delta
        if abs(delta) > tol:
            out["aug_status"] = "pending"; out["reject_or_pending_reason"] = f"aug_onset_disagree(>{tol})"
        else:
            out["aug_status"] = "use"; out["auto_augmented_onset_ok"] = True
    return out, aug.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", nargs="+", default=["fast", "normal", "slow"])
    ap.add_argument("--out-subdir", default="out")
    ap.add_argument("--per-origin", type=int, default=0,
                    help="0=각 origin에 전체 profile 1회씩(×len(profiles)). >0=round-robin 그만큼.")
    args = ap.parse_args()
    OUT = ROOT / "debug/dummy_gen" / args.out_subdir
    OUT.mkdir(parents=True, exist_ok=True)
    PROFILES = [{"body_size": b, "fall_speed": s} for b, s in product(BODIES, args.speeds)]
    origins = select_origins()
    mode = "all" if args.per_origin == 0 else f"roundrobin{args.per_origin}"
    n_per = len(PROFILES) if args.per_origin == 0 else args.per_origin
    print(f"[origin] {len(origins)} train non-WALK fall | speeds={args.speeds} profiles={len(PROFILES)} "
          f"mode={mode} → ×{n_per} = {len(origins)*n_per} | out={OUT.name}")
    tasks = []
    pc = 0
    for r in sorted(origins, key=lambda r: r["filename"]):
        on = int(float(r["onset_frame_clean"]))
        profs = PROFILES if args.per_origin == 0 else [PROFILES[(pc + k) % len(PROFILES)] for k in range(args.per_origin)]
        if args.per_origin:
            pc += args.per_origin
        for k, prof in enumerate(profs):
            seed = (abs(hash(r["filename"])) % (2**31)) + k * 7919
            tasks.append((r["filename"], on, r["subtype"], r["subject"], r["env"],
                          r["onset_status"], prof, k, seed))
    nw = min(16, 30)
    print(f"[생성] {len(tasks)} 더미, {nw} workers")
    rows, arrays, names = [], [], []
    done = 0
    import multiprocessing as mp
    nw = min(16, max(1, mp.cpu_count() - 2))
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futs = [pool.submit(_worker, t) for t in tasks]
        for fut in as_completed(futs):
            row, arr = fut.result()
            rows.append(row)
            if arr is not None:
                names.append(row["aug_filename"]); arrays.append(arr)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}]", flush=True)

    # lineage CSV
    cols = ["aug_filename", "origin_filename", "origin_split", "origin_subtype", "origin_subject",
            "origin_env", "aug_id", "transform_profile", "transform_params", "warp_factor",
            "scale_factor", "noise_snr", "crop_offset", "splice_boundaries_clean",
            "expected_onset_clean", "detected_onset_clean", "onset_delta", "baseline_noise_ratio",
            "auto_augmented_onset_ok", "aug_status", "reject_or_pending_reason"]
    with open(OUT / "lineage.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    # dummies npz (aug400 + filename, status, profile for downstream/validation)
    status_arr = {row["aug_filename"]: row["aug_status"] for row in rows}
    sub_arr = {row["aug_filename"]: row["origin_subtype"] for row in rows}
    np.savez_compressed(
        OUT / "dummies_clean400.npz",
        X=np.stack(arrays, axis=0) if arrays else np.empty((0, L.CLEAN_LEN, 104), np.float32),
        aug_filename=np.asarray(names, dtype=object),
        aug_status=np.asarray([status_arr[n] for n in names], dtype=object),
        origin_subtype=np.asarray([sub_arr[n] for n in names], dtype=object),
    )
    print(f"[생성] {OUT/'lineage.csv'} ({len(rows)}) | {OUT/'dummies_clean400.npz'} ({len(arrays)} arrays)")

    # 요약
    st = Counter(r["aug_status"] for r in rows)
    print(f"\n=== 분류: use {st.get('use',0)} / pending {st.get('pending',0)} / exclude {st.get('exclude',0)} ===")
    print(f"pending 사유: {dict(Counter(r['reject_or_pending_reason'] for r in rows if r['aug_status']=='pending'))}")
    print(f"profile 분포: {dict(Counter(r['transform_profile'] for r in rows))}")
    print(f"speed 분포: {dict(Counter(r['transform_profile'].split('_')[1] for r in rows))}")
    deltas = [int(r['onset_delta']) for r in rows if r['onset_delta'] != ""]
    if deltas:
        ad = np.abs(deltas)
        print(f"onset_delta(|·|): median={np.median(ad):.1f} p90={np.percentile(ad,90):.1f} max={ad.max()} (n={len(ad)})")
    use_by_speed = Counter(r['transform_profile'].split('_')[1] for r in rows if r['aug_status']=='use')
    print(f"use speed 분포: {dict(use_by_speed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
