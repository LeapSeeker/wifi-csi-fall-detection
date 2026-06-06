"""Gate 3 event-centered cache builder (read-only).

manifest_v2_manual_augmented 기준으로 fall 세션의 3종 crop cache 를 생성한다.
clean400 좌표계(Gate 1 확정: clean = A550[50:150]+[200:400]+[450:550]) 위에서 crop 후
동결 파이프라인(window_to_model_input = RPCA→ACF→SDP→global z-score)을 그대로 적용한다.

crop 3종 (모두 길이 300 → 길이효과 배제, 순수 정렬/post 효과):
  fixed         usable_for_fixed=True            clean[50:350]
  onset_primary usable_for_onset_aligned=True    clean[onset-50 : onset+250]
  onset_reduced usable_for_onset_aligned=True    clean[onset-100: onset+200]  (post amount ablation)
범위 밖이면 clamp 금지 → 생성 안 함, crop_exclude_reason=crop_out_of_bounds.
median fallback/imputation 금지: onset null 세션은 onset crop 생성 안 함.

제약(read-only): 원본 CSV·동결파일(pipeline/rpca/acf/sdp)·manifest 무수정. 산출은 cache 디렉토리.
test split 은 읽되(생성은 함) 설계/threshold/crop 통계 결정에 사용 금지 — 집계에서 분리 표기.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from model.preprocessing.loader import load_safesignal_csv          # noqa: E402
from model.preprocessing.resample import resample_to_100hz          # noqa: E402
from model.preprocessing.pipeline import window_to_model_input      # noqa: E402  (동결: RPCA→ACF→SDP→z)

FINAL = ROOT / "debug/modeling/diag_out/onset_detector/finalization"
V2 = FINAL / "manifest_v2_manual_augmented.csv"
CLEANED = ROOT / "data/cleaned"
OUTDIR = ROOT / "debug/modeling/diag_out/onset_detector/gate3_cache"

MIN_FRAMES = 550
WIN = 300
CLEAN_LEN = 400
MANIFEST_ID = "onset_manifest_v2_2026-06-06"
CLASSES = ("fall", "walking", "sit_stand", "lying", "standing", "picking")  # pretrained6, fall=0
WALK = {"FALL_WALK_F", "FALL_WALK_B"}
SUBTYPES = ["FALL_WALK_F", "FALL_WALK_B", "FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B"]
SPLITS = ["train", "val", "test", "out_of_scope"]
POLICIES = ["fixed", "onset_primary", "onset_reduced"]


def fnum(x):
    try:
        if x in ("", "None", None):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def build_clean400(a550):
    return np.concatenate([a550[50:150], a550[200:400], a550[450:550]], axis=0)  # (400, n_sc)


def crop_spec(policy, onset):
    if policy == "fixed":
        return 50, 350
    if onset is None:
        return None, None
    if policy == "onset_primary":
        return onset - 50, onset + 250
    if policy == "onset_reduced":
        return onset - 100, onset + 200
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="dry-run: 세션 수 제한")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not V2.exists():
        print(f"[중단] v2 manifest 없음: {V2}")
        return 1
    rows = list(csv.DictReader(open(V2, encoding="utf-8-sig")))
    if args.limit:
        rows = rows[:args.limit]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    store = {p: defaultdict(list) for p in POLICIES}   # policy -> field -> list
    index_rows = []                                     # crop_index.csv (모든 시도)
    t0 = time.time()
    n_load = 0
    aborts = []

    for i, m in enumerate(rows, 1):
        fn = m["filename"]
        subtype = m["subtype"]
        onset_status = m["onset_status"]
        uf = m["usable_for_fixed"] == "True"
        uoa = m["usable_for_onset_aligned"] == "True"
        onset = fnum(m["onset_frame_clean"])
        onset_i = int(onset) if onset is not None else None

        # 이 세션에서 시도할 policy 결정
        attempts = []  # (policy, cond_ok, start, end, base_reason)
        attempts.append(("fixed", uf, *crop_spec("fixed", onset_i), "" if uf else "not_usable_for_fixed"))
        for p in ("onset_primary", "onset_reduced"):
            if not uoa:
                attempts.append((p, False, None, None, "not_usable_for_onset_aligned"))
            elif onset_i is None:
                attempts.append((p, False, None, None, "onset_null"))
            else:
                s, e = crop_spec(p, onset_i)
                attempts.append((p, True, s, e, ""))

        need_data = any(c and (r == "") for (_p, c, _s, _e, r) in
                        [(p, c, s, e, ("oob" if (c and not (s is not None and 0 <= s and e <= CLEAN_LEN)) else ""))
                         for (p, c, s, e, _br) in attempts])
        # 데이터 로드 필요? (생성 가능한 crop 이 하나라도 있으면)
        load_needed = False
        for (p, c, s, e, br) in attempts:
            if c and s is not None and 0 <= s and e <= CLEAN_LEN:
                load_needed = True
        clean = None
        data_bad = False
        if load_needed:
            path = next(CLEANED.rglob(fn), None)
            if path is None:
                data_bad = True
                aborts.append((fn, "csv_not_found"))
            else:
                raw = load_safesignal_csv(path, rx="both")
                res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
                if int(res.resampled_count) < MIN_FRAMES:
                    data_bad = True
                    aborts.append((fn, f"resampled_lt_550({int(res.resampled_count)})"))
                else:
                    clean = build_clean400(res.amplitude[:MIN_FRAMES])
                    n_load += 1

        for (p, cond, s, e, base_reason) in attempts:
            reason = base_reason
            generated = False
            if cond and base_reason == "":
                if not (s is not None and 0 <= s and e <= CLEAN_LEN and (e - s) == WIN):
                    reason = "crop_out_of_bounds"
                elif data_bad or clean is None:
                    reason = "data_short_or_corrupt"
                else:
                    crop = clean[s:e]
                    x = window_to_model_input(crop)  # (1,28,20) 동결 파이프라인
                    generated = True
                    st = store[p]
                    st["X"].append(x.astype(np.float32))
                    st["y"].append(0)  # fall
                    st["filename"].append(fn)
                    st["env"].append(int(m["env"]))
                    st["subject"].append(int(m["subject"]))
                    st["subtype"].append(subtype)
                    st["split_assignment"].append(m["split_assignment"])
                    st["onset_frame_clean"].append(onset_i if onset_i is not None else -1)
                    st["crop_start_clean"].append(s)
                    st["crop_end_clean"].append(e)
                    st["crop_policy"].append(p)
                    st["usable_for_fixed"].append(uf)
                    st["usable_for_onset_aligned"].append(uoa)
                    st["exclude_scope"].append(m["exclude_scope"])
                    st["onset_status"].append(onset_status)
            index_rows.append({
                "filename": fn, "subtype": subtype, "env": m["env"], "subject": m["subject"],
                "split_assignment": m["split_assignment"], "onset_status": onset_status,
                "usable_for_fixed": uf, "usable_for_onset_aligned": uoa, "exclude_scope": m["exclude_scope"],
                "onset_frame_clean": onset_i if onset_i is not None else "",
                "crop_policy": p, "crop_start_clean": s if s is not None else "",
                "crop_end_clean": e if e is not None else "",
                "generated": generated,
                "crop_exclude_reason": "" if generated else (reason or "not_generated"),
                "manifest_id": MANIFEST_ID,
            })
        if i % 30 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] loaded={n_load} elapsed={time.time()-t0:.0f}s")

    # ── npz 저장 (policy별) ───────────────────────────────────────────────────
    if not args.overwrite:
        existing = [p for p in POLICIES if (OUTDIR / f"cache_{p}.npz").exists()]
        if existing:
            print(f"[중단] 기존 cache 존재: {existing} (--overwrite 필요)")
            return 1
    for p in POLICIES:
        st = store[p]
        n = len(st["X"])
        X = np.stack(st["X"], axis=0).astype(np.float32) if n else np.empty((0, 1, 28, 20), np.float32)
        np.savez_compressed(
            OUTDIR / f"cache_{p}.npz",
            X=X, y=np.asarray(st["y"], dtype=np.int64),
            filename=np.asarray(st["filename"], dtype=object),
            env=np.asarray(st["env"], dtype=np.int64),
            subject=np.asarray(st["subject"], dtype=np.int64),
            subtype=np.asarray(st["subtype"], dtype=object),
            split_assignment=np.asarray(st["split_assignment"], dtype=object),
            onset_frame_clean=np.asarray(st["onset_frame_clean"], dtype=np.int64),
            crop_start_clean=np.asarray(st["crop_start_clean"], dtype=np.int64),
            crop_end_clean=np.asarray(st["crop_end_clean"], dtype=np.int64),
            crop_policy=np.asarray(st["crop_policy"], dtype=object),
            usable_for_fixed=np.asarray(st["usable_for_fixed"], dtype=bool),
            usable_for_onset_aligned=np.asarray(st["usable_for_onset_aligned"], dtype=bool),
            exclude_scope=np.asarray(st["exclude_scope"], dtype=object),
            onset_status=np.asarray(st["onset_status"], dtype=object),
            source=np.asarray(["safesignal"] * n, dtype=object),
            classes=np.asarray(CLASSES, dtype=object),
            manifest_id=MANIFEST_ID, crop_policy_name=p,
        )
        print(f"[생성] {OUTDIR/f'cache_{p}.npz'} ({n} crops)")

    # crop_index.csv
    idx_cols = ["filename", "subtype", "env", "subject", "split_assignment", "onset_status",
                "usable_for_fixed", "usable_for_onset_aligned", "exclude_scope", "onset_frame_clean",
                "crop_policy", "crop_start_clean", "crop_end_clean", "generated", "crop_exclude_reason", "manifest_id"]
    with open(OUTDIR / "crop_index.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=idx_cols)
        w.writeheader()
        w.writerows(index_rows)
    print(f"[생성] {OUTDIR/'crop_index.csv'} ({len(index_rows)} rows)")

    # ── summary ───────────────────────────────────────────────────────────────
    def gen_rows(p):
        return [r for r in index_rows if r["crop_policy"] == p and r["generated"]]
    def oob_rows(p):
        return [r for r in index_rows if r["crop_policy"] == p and r["crop_exclude_reason"] == "crop_out_of_bounds"]

    def cnt_by(rows, key):
        return dict(Counter(r[key] for r in rows))

    fixed_gen = gen_rows("fixed")
    prim_gen = gen_rows("onset_primary")
    red_gen = gen_rows("onset_reduced")

    # paired: 같은 세션에서 fixed ∩ onset_primary 둘 다 생성
    fixed_fns = {r["filename"] for r in fixed_gen}
    prim_fns = {r["filename"] for r in prim_gen}
    paired_fns = fixed_fns & prim_fns
    def is_nonwalk_tv(fn):
        r = next(rr for rr in index_rows if rr["filename"] == fn)
        return r["subtype"] not in WALK and r["split_assignment"] in ("train", "val")
    paired_nonwalk_tv = [fn for fn in paired_fns if is_nonwalk_tv(fn)]
    paired_walk_tv = [fn for fn in paired_fns
                      if next(rr for rr in index_rows if rr["filename"] == fn)["subtype"] in WALK
                      and next(rr for rr in index_rows if rr["filename"] == fn)["split_assignment"] in ("train", "val")]

    def coverage(rows):
        walk = sum(1 for r in rows if r["subtype"] in WALK)
        return {"walk": walk, "non_walk": len(rows) - walk, "total": len(rows)}

    summary = {
        "meta": {"manifest_id": MANIFEST_ID, "sessions": len(rows), "loaded": n_load,
                 "window": WIN, "clean_len": CLEAN_LEN,
                 "pipeline": "clean400 concat (Gate1) → window_to_model_input(RPCA→ACF→SDP→z)",
                 "note_test": "test split 생성하되 설계/threshold/crop 통계 결정 미사용(분리표기)"},
        "generated": {p: len(gen_rows(p)) for p in POLICIES},
        "crop_out_of_bounds": {p: len(oob_rows(p)) for p in POLICIES},
        "onset_unusable": sum(1 for r in rows if r["onset_status"] == "onset_unusable"),
        "data_invalid": sum(1 for r in rows if r["onset_status"] == "data_invalid"),
        "fixed_by_split": cnt_by(fixed_gen, "split_assignment"),
        "fixed_by_subtype": cnt_by(fixed_gen, "subtype"),
        "onset_primary_by_split": cnt_by(prim_gen, "split_assignment"),
        "onset_primary_by_subtype": cnt_by(prim_gen, "subtype"),
        "onset_reduced_by_split": cnt_by(red_gen, "split_assignment"),
        "onset_reduced_by_subtype": cnt_by(red_gen, "subtype"),
        "coverage_fixed": coverage(fixed_gen),
        "coverage_onset_primary": coverage(prim_gen),
        "paired_fixed_and_onset_primary": {
            "total": len(paired_fns),
            "nonwalk_train_val": len(paired_nonwalk_tv),
            "walk_train_val": len(paired_walk_tv),
        },
        "aborts": aborts,
    }
    # env_subject별 생성 수 (fixed/primary)
    def env_sub(rows):
        return dict(Counter(f"E{r['env']}_S{int(r['subject']):02d}" for r in rows))
    summary["fixed_by_env_subject"] = env_sub(fixed_gen)
    summary["onset_primary_by_env_subject"] = env_sub(prim_gen)
    (OUTDIR / "gate3_cache_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # md
    def tier(n):
        return "결론금지(<10)" if n < 10 else ("exploratory(10-19)" if n < 20 else "제한적해석(20+)")
    L = ["# Gate 3 event-centered cache 요약\n",
         f"세션 {len(rows)} | clean400→window_to_model_input(동결) | window {WIN}.",
         "test split 은 생성하되 설계/threshold/crop 통계 결정 미사용.\n",
         "## crop 생성 수 / crop_out_of_bounds",
         "policy | generated | oob",
         "---|---|---"]
    for p in POLICIES:
        L.append(f"{p} | {len(gen_rows(p))} | {len(oob_rows(p))}")
    L.append(f"\nonset_unusable {summary['onset_unusable']} | data_invalid {summary['data_invalid']}")
    L.append("\n## fixed by split / subtype")
    L.append("split: " + ", ".join(f"{k} {v}" for k, v in summary["fixed_by_split"].items()))
    L.append("subtype: " + ", ".join(f"{st} {summary['fixed_by_subtype'].get(st,0)}" for st in SUBTYPES))
    L.append("\n## onset_primary by split / subtype")
    L.append("split: " + ", ".join(f"{k} {v}" for k, v in summary["onset_primary_by_split"].items()))
    L.append("subtype: " + ", ".join(f"{st} {summary['onset_primary_by_subtype'].get(st,0)}" for st in SUBTYPES))
    L.append("\n## onset_reduced by split / subtype")
    L.append("split: " + ", ".join(f"{k} {v}" for k, v in summary["onset_reduced_by_split"].items()))
    L.append("subtype: " + ", ".join(f"{st} {summary['onset_reduced_by_subtype'].get(st,0)}" for st in SUBTYPES))
    L.append("\n## coverage (inclusion rate)")
    L.append(f"- fixed: WALK {summary['coverage_fixed']['walk']} / non-WALK {summary['coverage_fixed']['non_walk']}")
    L.append(f"- onset_primary: WALK {summary['coverage_onset_primary']['walk']} / non-WALK {summary['coverage_onset_primary']['non_walk']}")
    L.append("\n## ★ paired comparison 가능 N (fixed ∩ onset_primary)")
    L.append(f"- 전체 paired: {len(paired_fns)}")
    L.append(f"- **non-WALK pooled train+val: {len(paired_nonwalk_tv)} → {tier(len(paired_nonwalk_tv))}** (항목4 메인)")
    L.append(f"- WALK pooled train+val: {len(paired_walk_tv)} → {tier(len(paired_walk_tv))} (exploratory)")
    L.append("\n## env×subject 생성 수 (fixed)")
    for es in sorted(summary["fixed_by_env_subject"]):
        L.append(f"- {es}: {summary['fixed_by_env_subject'][es]}")
    if aborts:
        L.append(f"\n## ⚠ aborts ({len(aborts)})")
        for fn, why in aborts:
            L.append(f"- {fn}: {why}")
    (OUTDIR / "gate3_cache_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[생성] {OUTDIR}/ (cache_*.npz, crop_index.csv, gate3_cache_summary.json/md) | {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
