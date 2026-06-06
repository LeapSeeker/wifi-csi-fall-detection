"""4-arm 더미 학습용 증강 cache 빌드 + sanity (read-only 입력).

C fixed_augmented      = item4_cache_fixed         + 더미 fixed crop(clean[50:350])
D onset_augmented      = item4_cache_onset_primary + 더미 onset crop(clean[det-50:det+250])
더미 = out2 use 301 (slow 제외, onset_delta 검증 통과). train origin 기반 → train 에만 추가.

sanity: ① 더미 origin 전부 train ② origin이 val/test 파생 아님 ③ val/test 에 더미 없음.
산출: item4_cache_fixed_aug.npz, item4_cache_onset_primary_aug.npz. class ratio 변화 보고.
read-only: 원본 cache·manifest·동결파일 무수정.
"""
from __future__ import annotations
import csv
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ITEM4 = ROOT / "debug/modeling/diag_out/onset_detector/item4"
OUT2 = ROOT / "debug/dummy_gen/out2"
V2 = ROOT / "debug/modeling/diag_out/onset_detector/finalization/manifest_v2_manual_augmented.csv"
WIN = 300


def _feat(args):
    arr, s, e = args
    from model.preprocessing.pipeline import window_to_model_input
    return window_to_model_input(arr[s:e].astype(np.float64)).astype(np.float32)


def main():
    man = {r["filename"]: r for r in csv.DictReader(open(V2, encoding="utf-8-sig"))}
    lin = {r["aug_filename"]: r for r in csv.DictReader(open(OUT2 / "lineage.csv", encoding="utf-8-sig"))}
    dz = np.load(OUT2 / "dummies_clean400.npz", allow_pickle=True)
    names = [str(x) for x in dz["aug_filename"]]
    status = {n: str(s) for n, s in zip(names, dz["aug_status"])}
    Xd = dz["X"]
    idx_of = {n: i for i, n in enumerate(names)}

    use = [n for n in names if status[n] == "use"]
    print(f"[더미] use {len(use)} / 전체 {len(names)}")

    # ── sanity ────────────────────────────────────────────────────────────────
    bad_split = [n for n in use if man.get(lin[n]["origin_filename"], {}).get("split_assignment") != "train"]
    print(f"[sanity ①②] use 더미 origin split=train 위반: {len(bad_split)} {'OK' if not bad_split else bad_split[:3]}")
    # origin 분포
    print(f"  origin subtype: {dict(Counter(lin[n]['origin_subtype'] for n in use))}")

    # ── base cache 로드 ───────────────────────────────────────────────────────
    base_fixed = np.load(ITEM4 / "item4_cache_fixed.npz", allow_pickle=True)
    base_onset = np.load(ITEM4 / "item4_cache_onset_primary.npz", allow_pickle=True)
    # sanity ③: base val/test 에 aug_filename 없음(원본만)
    base_fns = set(str(x) for x in base_fixed["filename"]) | set(str(x) for x in base_onset["filename"])
    leak = [n for n in use if n in base_fns]
    print(f"[sanity ③] 더미가 기존 cache(원본)에 이미 존재: {len(leak)} {'OK(없음)' if not leak else leak[:3]}")

    # ── 더미 crop feature 계산 (병렬) ────────────────────────────────────────
    fixed_tasks, fixed_meta = [], []
    onset_tasks, onset_meta = [], []
    oob_onset = 0
    for n in use:
        arr = Xd[idx_of[n]]
        o = lin[n]
        meta = {"filename": n, "subject": int(o["origin_subject"]), "env": int(o["origin_env"]),
                "subtype": o["origin_subtype"]}
        fixed_tasks.append((arr, 50, 350)); fixed_meta.append(meta)
        det = o["detected_onset_clean"]
        if det not in ("", "None"):
            on = int(float(det))
            s, e = on - 50, on + 250
            if 0 <= s and e <= 400:
                onset_tasks.append((arr, s, e)); onset_meta.append(meta)
            else:
                oob_onset += 1
        else:
            oob_onset += 1
    print(f"[crop] fixed {len(fixed_tasks)} | onset {len(onset_tasks)} (onset OOB/missing {oob_onset})")

    import multiprocessing as mp
    nw = min(16, max(1, mp.cpu_count() - 2))

    def compute(tasks):
        feats = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=nw) as pool:
            futs = {pool.submit(_feat, t): i for i, t in enumerate(tasks)}
            done = 0
            for fut in as_completed(futs):
                feats[futs[fut]] = fut.result()
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    print(f"    feat {done}/{len(tasks)}", flush=True)
        return feats

    print("[feature] fixed crops..."); ff = compute(fixed_tasks)
    print("[feature] onset crops..."); of = compute(onset_tasks)

    # ── cache 합성 ────────────────────────────────────────────────────────────
    def build(base, feats, metas, out_name):
        n = len(feats)
        X = np.concatenate([base["X"].astype(np.float32), np.stack(feats).astype(np.float32)], axis=0)
        def cat(key, vals):
            return np.concatenate([base[key], np.asarray(vals, dtype=base[key].dtype if base[key].dtype != object else object)])
        y = np.concatenate([base["y"].astype(np.int64), np.zeros(n, np.int64)])
        out = {
            "X": X, "y": y,
            "subject": np.concatenate([base["subject"].astype(np.int64), np.asarray([m["subject"] for m in metas], np.int64)]),
            "source": np.concatenate([base["source"], np.asarray(["safesignal_dummy"] * n, object)]),
            "env": np.concatenate([base["env"].astype(np.int64), np.asarray([m["env"] for m in metas], np.int64)]),
            "filename": np.concatenate([base["filename"], np.asarray([m["filename"] for m in metas], object)]),
            "activity": np.concatenate([base["activity"], np.asarray([m["subtype"] for m in metas], object)]),
            "trial": np.concatenate([base["trial"].astype(np.int64), np.full(n, -1, np.int64)]),
            "classes": base["classes"], "class_policy": "pretrained6",
            "is_augmented": np.concatenate([base["is_augmented"].astype(bool), np.ones(n, bool)]),
            "split_assignment": np.concatenate([base["split_assignment"], np.asarray(["train"] * n, object)]),
        }
        np.savez_compressed(ITEM4 / out_name, **out)
        # class ratio (train)
        sp = out["split_assignment"]; yy = out["y"]
        tr = sp == "train"
        fall_tr = int(((yy == 0) & tr).sum()); non_tr = int(((yy != 0) & tr).sum())
        b_tr = base["split_assignment"] == "train"
        bf = int(((base["y"] == 0) & b_tr).sum())
        print(f"[생성] {out_name}: +{n} 더미 | train fall {bf}→{fall_tr} / non-fall {non_tr} "
              f"(fall:non = {fall_tr/non_tr:.3f}, 원본 {bf/non_tr:.3f})")
        return fall_tr, non_tr, bf

    print()
    build(base_fixed, ff, fixed_meta, "item4_cache_fixed_aug.npz")
    build(base_onset, of, onset_meta, "item4_cache_onset_primary_aug.npz")
    print("\n[완료] C=fixed_aug, D=onset_primary_aug 생성. 더미 train-only, val/test 봉인 유지.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
