"""항목4: policy별 SafeSignal 학습 cache 조립 (read-only 입력).

fall 행만 event-centered crop(fixed/onset_primary/onset_reduced)으로 교체하고 non-fall 은
기존 sliding 그대로 유지(D-031 Q3). split 은 **frozen** — policy 간 동일해야 paired 비교가
성립하므로 per-policy 재계산 금지:
  - fall 세션 split = v2 manifest split_assignment (authoritative)
  - non-fall 세션 split = 원본 cache 에 canonical within_subject seed42 splitter 1회 적용 (frozen)
  - fall 에 대해 manifest vs canonical splitter 일치율 보고(투명성)
산출: item4/item4_cache_{fixed,onset_primary,onset_reduced}.npz (split_assignment 컬럼 포함)

제약(read-only): 원본 cache·gate3 cache·manifest·train.py(동결) 무수정. 신규 cache 만 생성.
"""
from __future__ import annotations
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from model.finetune.train import split_safesignal_within_subject  # noqa: E402


class _SplitShim:
    """split_safesignal_within_subject 가 쓰는 최소 인터페이스(.filenames/.subjects/.y)만 제공.
    (Subset 은 .indices 만 사용하므로 __getitem__ 불필요 — 클래스 remap 회피.)"""
    def __init__(self, npz):
        self.filenames = [str(x) for x in npz["filename"]]
        self.subjects = [int(x) for x in npz["subject"]]
        self.y = npz["y"].astype("int64")

ORIG = ROOT / "model/finetune/cache/safesignal_e1234_pretrained6.npz"
GATE3 = ROOT / "debug/modeling/diag_out/onset_detector/gate3_cache"
V2 = ROOT / "debug/modeling/diag_out/onset_detector/finalization/manifest_v2_manual_augmented.csv"
OUTDIR = ROOT / "debug/modeling/diag_out/onset_detector/item4"
POLICIES = ["fixed", "onset_primary", "onset_reduced"]
SEED, VAL_R, TEST_R = 42, 0.2, 0.2
FALL_IDX = 0


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    orig = np.load(ORIG, allow_pickle=True)
    o_fn = np.array([str(x) for x in orig["filename"]])
    o_y = orig["y"].astype(np.int64)
    o_split_fields = {k: orig[k] for k in orig.files}

    # ── canonical split (frozen, 전 세션) ─────────────────────────────────────
    ds = _SplitShim(orig)
    tr, va, te = split_safesignal_within_subject(ds, val_ratio=VAL_R, test_ratio=TEST_R, seed=SEED)
    split_map = {}
    for sub, name in ((tr, "train"), (va, "val"), (te, "test")):
        for idx in sub.indices:
            split_map[str(ds.filenames[idx])] = name
    print(f"[canonical split] sessions={len(split_map)} "
          f"({dict(Counter(split_map.values()))})")

    # ── v2 manifest (fall) split + onset ──────────────────────────────────────
    man = {r["filename"]: r for r in csv.DictReader(open(V2, encoding="utf-8-sig"))}

    # fall split 일치율 (manifest vs canonical) — fall 은 manifest 를 authoritative 로 사용
    fall_fns = sorted(set(o_fn[o_y == FALL_IDX].tolist()))
    agree = miss = mismatch = 0
    for fn in fall_fns:
        msp = man.get(fn, {}).get("split_assignment")
        csp = split_map.get(fn)
        if msp is None:
            miss += 1
        elif msp == csp:
            agree += 1
        else:
            mismatch += 1
    print(f"[fall split 일치] manifest∩cache fall={len(fall_fns)} | agree {agree} / mismatch {mismatch} / not_in_manifest {miss}")

    # frozen split: fall→manifest, non-fall→canonical
    def split_of(fn, is_fall):
        if is_fall and fn in man and man[fn]["split_assignment"]:
            return man[fn]["split_assignment"]
        return split_map.get(fn, "train")

    # ── non-fall 행 (원본 그대로 유지) ────────────────────────────────────────
    nonfall_mask = o_y != FALL_IDX
    nf_idx = np.where(nonfall_mask)[0]
    nf_split = np.array([split_of(o_fn[i], False) for i in nf_idx], dtype=object)
    print(f"[non-fall] {len(nf_idx)} windows ({dict(Counter(nf_split.tolist()))})")

    # ── policy별 cache 조립 ───────────────────────────────────────────────────
    classes = list(orig["classes"])
    for policy in POLICIES:
        g = np.load(GATE3 / f"cache_{policy}.npz", allow_pickle=True)
        gX = g["X"].astype(np.float32)
        gfn = np.array([str(x) for x in g["filename"]])
        g_n = len(gX)
        g_split = np.array([man[fn]["split_assignment"] if fn in man and man[fn]["split_assignment"]
                            else split_map.get(fn, "train") for fn in gfn], dtype=object)
        g_env = g["env"].astype(np.int64)
        g_subj = g["subject"].astype(np.int64)
        g_sub = np.array([str(x) for x in g["subtype"]], dtype=object)

        # concat: non-fall(원본) + fall crop(gate3)
        X = np.concatenate([orig["X"][nf_idx].astype(np.float32), gX], axis=0)
        y = np.concatenate([o_y[nf_idx], np.zeros(g_n, dtype=np.int64)])
        filename = np.concatenate([o_fn[nf_idx], gfn]).astype(object)
        subject = np.concatenate([orig["subject"][nf_idx].astype(np.int64), g_subj])
        env = np.concatenate([orig["env"][nf_idx].astype(np.int64), g_env])
        activity = np.concatenate([np.array([str(x) for x in orig["activity"][nf_idx]], dtype=object), g_sub]).astype(object)
        trial = np.concatenate([orig["trial"][nf_idx].astype(np.int64), np.full(g_n, -1, dtype=np.int64)])
        source = np.array(["safesignal"] * len(y), dtype=object)
        is_aug = np.zeros(len(y), dtype=bool)
        split_assignment = np.concatenate([nf_split, g_split]).astype(object)

        out = OUTDIR / f"item4_cache_{policy}.npz"
        np.savez_compressed(
            out, X=X, y=y, subject=subject, source=source, env=env,
            filename=filename, activity=activity, trial=trial,
            classes=np.asarray(classes, dtype=object), class_policy="pretrained6",
            is_augmented=is_aug, split_assignment=split_assignment, crop_policy=policy,
        )
        fall_n = int((y == FALL_IDX).sum())
        sp_fall = dict(Counter(split_assignment[i] for i in range(len(y)) if y[i] == FALL_IDX))
        print(f"[생성] {out.name}: total {len(y)} (non-fall {len(nf_idx)} + fall {fall_n}) | "
              f"fall split {sp_fall}")

    print("\n[완료] item4 policy caches. split frozen (fall=manifest, non-fall=canonical seed42).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
