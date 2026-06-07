"""balanced 증강 cache 빌더 (read-only 입력) — fall 더미 + non-fall 더미 함께 append.

기존 fall-only 비통제 cache(item4_cache_*_aug.npz)의 class-ratio artifact 통제를 위해,
기존 fall 더미 **+ 신규 non-fall 더미(C/D 공유 단일 set)**를 함께 추가한 balanced cache 생성.

C item4_cache_fixed_balanced_aug         = fixed base + fall 더미 fixed crop(clean[50:350]) + non-fall shared
D item4_cache_onset_primary_balanced_aug = onset base + fall 더미 onset crop(clean[det-50:det+250]) + non-fall shared

fall 더미 crop 계산은 item4_build_arm_caches.py 와 1:1 동일(동일 use 301, 동일 crop, 동일
window_to_model_input) → 기존 fall-only 실험의 fall 더미 부분과 일치. non-fall shared set 은
nonfall_dummies.npz(이미 (N,1,28,20)) 를 두 arm 에 동일하게 append.

assert: ① 더미 split 전부 train ② val/test 더미 0 ③ fall 더미 origin split=train(v2)
④ non-fall 더미 origin split=train(lineage) ⑤ C/D non-fall 더미 filename set 동일.

제약(read-only): 원본 cache·manifest·동결 파일·기존 *_aug.npz 무수정/덮어쓰기 금지. _balanced_aug 신규.
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
BAL = ROOT / "debug/dummy_gen/balanced"
V2 = ROOT / "debug/modeling/diag_out/onset_detector/finalization/manifest_v2_manual_augmented.csv"
WIN = 300

BASE_FIXED = ITEM4 / "item4_cache_fixed.npz"
BASE_ONSET = ITEM4 / "item4_cache_onset_primary.npz"
FALL_NPZ = OUT2 / "dummies_clean400.npz"
FALL_LIN = OUT2 / "lineage.csv"
NF_NPZ = BAL / "nonfall_dummies.npz"
NF_LIN = BAL / "nonfall_lineage.csv"
OUT_FIXED = ITEM4 / "item4_cache_fixed_balanced_aug.npz"
OUT_ONSET = ITEM4 / "item4_cache_onset_primary_balanced_aug.npz"


def preflight():
    req = [BASE_FIXED, BASE_ONSET, FALL_NPZ, FALL_LIN, NF_NPZ, NF_LIN, V2]
    miss = [p for p in req if not p.exists()]
    if miss:
        print("[FAIL-FAST] 필수 입력 없음:")
        for p in miss:
            tip = "  ← Drive/백업에서 복원 필요" if p == FALL_NPZ else ""
            print(f"  - {p}{tip}")
        print("특히 dummies_clean400.npz 없으면 balanced cache 진행 금지(스펙 0-1).")
        raise SystemExit(2)


def _feat(args):
    arr, s, e = args
    from model.preprocessing.pipeline import window_to_model_input
    return window_to_model_input(arr[s:e].astype(np.float64)).astype(np.float32)


def _compute(tasks, nw, tag):
    feats = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futs = {pool.submit(_feat, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            feats[futs[fut]] = fut.result(); done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"    {tag} feat {done}/{len(tasks)}", flush=True)
    return feats


def main():
    preflight()
    import multiprocessing as mp
    nw = min(16, max(1, mp.cpu_count() - 2))

    man = {r["filename"]: r for r in csv.DictReader(open(V2, encoding="utf-8-sig"))}
    lin = {r["aug_filename"]: r for r in csv.DictReader(open(FALL_LIN, encoding="utf-8-sig"))}
    dz = np.load(FALL_NPZ, allow_pickle=True)
    names = [str(x) for x in dz["aug_filename"]]
    status = {n: str(s) for n, s in zip(names, dz["aug_status"])}
    Xd = dz["X"]
    idx_of = {n: i for i, n in enumerate(names)}
    use = [n for n in names if status[n] == "use"]
    print(f"[fall 더미] use {len(use)} / 전체 {len(names)}")

    # assert ③: fall 더미 origin split=train (v2 manifest)
    bad = [n for n in use if man.get(lin[n]["origin_filename"], {}).get("split_assignment") != "train"]
    assert not bad, f"fall 더미 origin split!=train: {bad[:5]}"
    print(f"[assert ③] fall 더미 origin split=train OK | origin subtype "
          f"{dict(Counter(lin[n]['origin_subtype'] for n in use))}")

    # ── non-fall 더미 로드 (이미 (N,1,28,20)) ─────────────────────────────────
    nf = np.load(NF_NPZ, allow_pickle=True)
    nfX = nf["X"].astype(np.float32)
    nf_fn = np.asarray([str(x) for x in nf["filename"]], object)
    nf_y = nf["y"].astype(np.int64)
    nf_subj = nf["subject"].astype(np.int64)
    nf_env = nf["env"].astype(np.int64)
    nf_act = np.asarray([str(x) for x in nf["activity"]], object)
    nf_trial = nf["trial"].astype(np.int64)
    print(f"[non-fall 더미] {len(nfX)} (shared) | class "
          f"{dict(Counter(int(c) for c in nf_y))}")
    # assert ④: non-fall 더미 origin split=train (lineage)
    nflin = {r["aug_filename"]: r for r in csv.DictReader(open(NF_LIN, encoding="utf-8-sig"))}
    bad_nf = [str(n) for n in nf_fn if nflin.get(str(n), {}).get("origin_split") != "train"]
    assert not bad_nf, f"non-fall 더미 origin split!=train: {bad_nf[:5]}"
    print(f"[assert ④] non-fall 더미 origin split=train OK")

    # ── base cache ────────────────────────────────────────────────────────────
    base_fixed = np.load(BASE_FIXED, allow_pickle=True)
    base_onset = np.load(BASE_ONSET, allow_pickle=True)
    base_fns = set(str(x) for x in base_fixed["filename"]) | set(str(x) for x in base_onset["filename"])
    leak = [n for n in use if n in base_fns] + [str(n) for n in nf_fn if str(n) in base_fns]
    assert not leak, f"더미가 base(원본) cache 에 이미 존재: {leak[:5]}"
    print(f"[assert leak] 더미∩base 원본 없음 OK")

    # ── fall 더미 arm별 crop feature ──────────────────────────────────────────
    fixed_tasks, fixed_meta, onset_tasks, onset_meta = [], [], [], []
    oob = 0
    for n in use:
        arr = Xd[idx_of[n]]; o = lin[n]
        meta = {"filename": n, "subject": int(o["origin_subject"]), "env": int(o["origin_env"]),
                "subtype": o["origin_subtype"]}
        fixed_tasks.append((arr, 50, 350)); fixed_meta.append(meta)
        det = o["detected_onset_clean"]
        if det not in ("", "None"):
            on = int(float(det)); s, e = on - 50, on + 250
            if 0 <= s and e <= 400:
                onset_tasks.append((arr, s, e)); onset_meta.append(meta)
            else:
                oob += 1
        else:
            oob += 1
    print(f"[fall crop] fixed {len(fixed_tasks)} | onset {len(onset_tasks)} (onset OOB/missing {oob})")
    print("[feature] fall fixed crops..."); ff = _compute(fixed_tasks, nw, "fixed")
    print("[feature] fall onset crops..."); of = _compute(onset_tasks, nw, "onset")

    # ── cache 합성 (base + fall 더미 crop + non-fall shared) ───────────────────
    def build(base, fall_feats, fall_metas, out_path, crop_policy):
        nfeat = [np.stack(fall_feats).astype(np.float32)] if fall_feats else []
        n_fall = len(fall_feats); n_nf = len(nfX)
        X = np.concatenate([base["X"].astype(np.float32)] + nfeat + [nfX], axis=0)
        # fall 더미 메타
        f_subj = np.asarray([m["subject"] for m in fall_metas], np.int64)
        f_env = np.asarray([m["env"] for m in fall_metas], np.int64)
        f_fn = np.asarray([m["filename"] for m in fall_metas], object)
        f_act = np.asarray([m["subtype"] for m in fall_metas], object)
        out = {
            "X": X,
            "y": np.concatenate([base["y"].astype(np.int64), np.zeros(n_fall, np.int64), nf_y]),
            "subject": np.concatenate([base["subject"].astype(np.int64), f_subj, nf_subj]),
            "source": np.concatenate([base["source"],
                                      np.asarray(["safesignal_dummy"] * (n_fall + n_nf), object)]),
            "env": np.concatenate([base["env"].astype(np.int64), f_env, nf_env]),
            "filename": np.concatenate([base["filename"], f_fn, nf_fn]).astype(object),
            "activity": np.concatenate([base["activity"], f_act, nf_act]).astype(object),
            "trial": np.concatenate([base["trial"].astype(np.int64),
                                     np.full(n_fall, -1, np.int64), nf_trial]),
            "classes": base["classes"], "class_policy": "pretrained6",
            "is_augmented": np.concatenate([base["is_augmented"].astype(bool),
                                            np.ones(n_fall + n_nf, bool)]),
            "split_assignment": np.concatenate([base["split_assignment"],
                                                np.asarray(["train"] * (n_fall + n_nf), object)]),
            "crop_policy": str(base["crop_policy"]) if "crop_policy" in base.files else crop_policy,
        }
        # ── assert ①②: 더미 split 전부 train / val·test 더미 0 ─────────────────
        n_base = len(base["y"]); dummy_split = out["split_assignment"][n_base:]
        assert all(s == "train" for s in dummy_split), "더미 split!=train 존재"
        assert int(np.sum((out["split_assignment"] != "train") & out["is_augmented"])) == 0, \
            "val/test 더미 존재"
        np.savez_compressed(out_path, **out)
        sp = out["split_assignment"]; yy = out["y"]; tr = sp == "train"
        fall_tr = int(((yy == 0) & tr).sum()); non_tr = int(((yy != 0) & tr).sum())
        b_tr = base["split_assignment"] == "train"; bf = int(((base["y"] == 0) & b_tr).sum())
        bnf = int(((base["y"] != 0) & b_tr).sum())
        print(f"[생성] {out_path.name}: +{n_fall} fall더미 +{n_nf} nonfall더미 | "
              f"train fall {bf}→{fall_tr} / non-fall {bnf}→{non_tr} "
              f"(fall:non {fall_tr/non_tr:.3f}, 원본 {bf/bnf:.3f})")
        return out

    print()
    out_c = build(base_fixed, ff, fixed_meta, OUT_FIXED, "fixed")
    out_d = build(base_onset, of, onset_meta, OUT_ONSET, "onset_primary")

    # ── assert ⑤: C/D non-fall 더미 filename set 동일 ─────────────────────────
    def nf_set(out):
        return set(str(f) for f, a in zip(out["filename"], out["is_augmented"])
                   if a and str(f) in set(nf_fn.tolist()))
    cset, dset = nf_set(out_c), nf_set(out_d)
    assert cset == dset, f"C/D non-fall 더미 filename set 불일치: only_C {list(cset-dset)[:3]} only_D {list(dset-cset)[:3]}"
    print(f"[assert ⑤] C/D non-fall 더미 filename set 동일 OK ({len(cset)})")
    print("\n[완료] C=fixed_balanced_aug, D=onset_primary_balanced_aug. fall+non-fall 더미 train-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
