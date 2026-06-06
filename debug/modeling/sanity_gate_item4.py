"""항목4 Sanity gate (read-only). GPU 학습 전 필수 통과 검증.

검사:
  1. cache class label 분포 (3 cache 모두 fall-only 여야)
  2. split leakage 없음 — 같은 filename 이 2개 split 에 들어가지 않음 (세션 단위)
  3. train/val/test 중복 없음 (filename 기준, cache 간 split 일관)
  4. fixed ∩ onset_primary paired key 가 session(filename) 단위로 정확히 매칭
     (onset_primary ⊆ fixed, 1세션 1crop, subtype/split 동일)
  5. crop 범위 무결성 (fixed=[50,350], 길이 300, onset crop in [0,400], onset 정렬 일치)
  6. X 무결성 (NaN/inf 없음, shape (N,1,28,20))
  7. seed/config 재현성 기록

입력: gate3_cache/cache_*.npz, crop_index.csv, manifest_v2. 산출 없음(보고 + md).
PASS/FAIL 종합 후 FAIL 이면 GPU 학습 금지 신호.
"""
from __future__ import annotations
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GATE3 = ROOT / "debug/modeling/diag_out/onset_detector/gate3_cache"
FINAL = ROOT / "debug/modeling/diag_out/onset_detector/finalization"
V2 = FINAL / "manifest_v2_manual_augmented.csv"
OUT_MD = GATE3 / "sanity_gate_item4.md"
POLICIES = ["fixed", "onset_primary", "onset_reduced"]
WALK = {"FALL_WALK_F", "FALL_WALK_B"}

# 재현성 설정 (기록용)
SPLIT_SEED = "within_subject_seed42_val0.2_test0.2_pretrained6"  # manifest split_id (동결)
TRAIN_SEED = 42  # 학습 seed — run config 에 고정 기록 필요
CONFIG = {"class_policy": "pretrained6", "pipeline": "RPCA→ACF→SDP→global z-score",
          "window": 300, "clean_coord": "clean400 concat (Gate1)", "fall_label_index": 0}


def main():
    checks = []  # (name, ok, detail)

    def load(p):
        return np.load(GATE3 / f"cache_{p}.npz", allow_pickle=True)

    caches = {p: load(p) for p in POLICIES}
    ci = list(csv.DictReader(open(GATE3 / "crop_index.csv", encoding="utf-8-sig")))
    man = {r["filename"]: r for r in csv.DictReader(open(V2, encoding="utf-8-sig"))}

    L = ["# 항목4 Sanity gate 결과 (read-only)\n"]

    # ── 1. label 분포 ─────────────────────────────────────────────────────────
    L.append("## 1. cache class label 분포 (fall-only 기대)")
    L.append("policy | N | y 분포 | classes[fall_idx]")
    L.append("---|---|---|---")
    ok1 = True
    for p in POLICIES:
        d = caches[p]
        y = d["y"]
        cls = list(d["classes"])
        dist = dict(Counter(int(v) for v in y))
        allfall = set(dist) == {0} if len(y) else True
        ok1 = ok1 and allfall and cls[0] == "fall"
        L.append(f"{p} | {len(y)} | {dist} | {cls[0]}")
    checks.append(("1.label fall-only", ok1, "모든 crop y=0(fall), classes[0]=fall"))

    # ── 2&3. split leakage / 중복 ─────────────────────────────────────────────
    # filename -> set of splits (cache 전반 + manifest)
    fn_split = defaultdict(set)
    for p in POLICIES:
        d = caches[p]
        for fn, sp in zip(d["filename"], d["split_assignment"]):
            fn_split[str(fn)].add(str(sp))
    multi = {fn: s for fn, s in fn_split.items() if len(s) > 1}
    # cache split == manifest split ?
    mism = []
    for fn, s in fn_split.items():
        msp = man.get(fn, {}).get("split_assignment")
        if msp and {msp} != s:
            mism.append((fn, list(s), msp))
    ok2 = (len(multi) == 0)
    ok3 = (len(mism) == 0)
    L.append("\n## 2. split leakage (같은 filename 이 복수 split)")
    L.append(f"- 복수 split filename: **{len(multi)}** {'(없음 ✓)' if not multi else dict(list(multi.items())[:5])}")
    checks.append(("2.no split leakage", ok2, f"{len(multi)} sessions in >1 split"))
    L.append("## 3. cache split == manifest split (일관)")
    L.append(f"- 불일치: **{len(mism)}** {'(없음 ✓)' if not mism else mism[:5]}")
    checks.append(("3.cache/manifest split 일치", ok3, f"{len(mism)} mismatches"))

    # split 분포
    split_by_policy = {p: dict(Counter(str(s) for s in caches[p]["split_assignment"])) for p in POLICIES}
    L.append("\nsplit 분포:")
    for p in POLICIES:
        L.append(f"- {p}: {split_by_policy[p]}")

    # ── 4. paired key 매칭 ────────────────────────────────────────────────────
    def fns(p):
        return [str(x) for x in caches[p]["filename"]]
    fixed_fns = set(fns("fixed"))
    prim_fns = set(fns("onset_primary"))
    red_fns = set(fns("onset_reduced"))
    # 1세션 1crop (중복 없음)
    dup_fixed = [fn for fn, c in Counter(fns("fixed")).items() if c > 1]
    dup_prim = [fn for fn, c in Counter(fns("onset_primary")).items() if c > 1]
    prim_not_in_fixed = prim_fns - fixed_fns
    red_not_in_fixed = red_fns - fixed_fns
    # subtype/split 동일성 (paired)
    sub_of = {}
    for p in POLICIES:
        for fn, st, sp in zip(fns(p), caches[p]["subtype"], caches[p]["split_assignment"]):
            sub_of.setdefault(fn, (str(st), str(sp)))
    paired = fixed_fns & prim_fns
    paired_meta_ok = all(
        (str(caches["fixed"]["subtype"][list(fns("fixed")).index(fn)]) ==
         str(caches["onset_primary"]["subtype"][list(fns("onset_primary")).index(fn)]))
        for fn in list(paired)[:0]  # subtype 동일은 아래서 별도 검증(간단화)
    ) if False else True
    ok4 = (len(dup_fixed) == 0 and len(dup_prim) == 0 and len(prim_not_in_fixed) == 0)
    L.append("\n## 4. paired key 매칭 (fixed ∩ onset_primary, session 단위)")
    L.append(f"- 1세션 1crop: fixed dup {len(dup_fixed)}, onset_primary dup {len(dup_prim)} {'✓' if not (dup_fixed or dup_prim) else '✗'}")
    L.append(f"- onset_primary ⊆ fixed: 미포함 {len(prim_not_in_fixed)} {'(전부 fixed에 존재 ✓)' if not prim_not_in_fixed else list(prim_not_in_fixed)[:5]}")
    L.append(f"- onset_reduced ⊆ fixed: 미포함 {len(red_not_in_fixed)} {'✓' if not red_not_in_fixed else list(red_not_in_fixed)[:5]}")
    L.append(f"- paired(fixed∩onset_primary) = **{len(paired)}** (= onset_primary {len(prim_fns)} 와 일치 {'✓' if len(paired)==len(prim_fns) else '✗'})")
    checks.append(("4.paired key 매칭", ok4, f"dup {len(dup_fixed)}/{len(dup_prim)}, prim⊄fixed {len(prim_not_in_fixed)}"))

    # subtype 동일성 검증 (paired 세션)
    fixed_idx = {fn: i for i, fn in enumerate(fns("fixed"))}
    prim_idx = {fn: i for i, fn in enumerate(fns("onset_primary"))}
    sub_mismatch = [fn for fn in paired
                    if str(caches["fixed"]["subtype"][fixed_idx[fn]]) != str(caches["onset_primary"]["subtype"][prim_idx[fn]])
                    or str(caches["fixed"]["split_assignment"][fixed_idx[fn]]) != str(caches["onset_primary"]["split_assignment"][prim_idx[fn]])]
    ok4b = len(sub_mismatch) == 0
    L.append(f"- paired subtype/split 동일: 불일치 {len(sub_mismatch)} {'✓' if ok4b else sub_mismatch[:5]}")
    checks.append(("4b.paired subtype/split 동일", ok4b, f"{len(sub_mismatch)} mismatch"))

    # ── 5. crop 범위 무결성 ───────────────────────────────────────────────────
    bad_range = []
    for p in POLICIES:
        d = caches[p]
        for fn, s, e, on in zip(d["filename"], d["crop_start_clean"], d["crop_end_clean"],
                                d["onset_frame_clean"]):
            s, e, on = int(s), int(e), int(on)
            if e - s != 300 or s < 0 or e > 400:
                bad_range.append((p, str(fn), s, e))
            if p == "fixed" and (s, e) != (50, 350):
                bad_range.append((p, str(fn), s, e))
            if p == "onset_primary" and (s != on - 50 or e != on + 250):
                bad_range.append((p, str(fn), s, e, on))
            if p == "onset_reduced" and (s != on - 100 or e != on + 200):
                bad_range.append((p, str(fn), s, e, on))
    ok5 = len(bad_range) == 0
    L.append("\n## 5. crop 범위 무결성 (길이300, [0,400], 정렬 일치, clamp 없음)")
    L.append(f"- 위반: **{len(bad_range)}** {'(없음 ✓)' if ok5 else bad_range[:5]}")
    checks.append(("5.crop 범위 무결성", ok5, f"{len(bad_range)} violations"))

    # ── 6. X 무결성 ──────────────────────────────────────────────────────────
    ok6 = True
    L.append("\n## 6. X 텐서 무결성")
    for p in POLICIES:
        X = caches[p]["X"]
        shape_ok = (X.ndim == 4 and X.shape[1:] == (1, 28, 20))
        finite = bool(np.isfinite(X).all()) if X.size else True
        ok6 = ok6 and shape_ok and finite
        L.append(f"- {p}: shape {X.shape} {'✓' if shape_ok else '✗'} | finite {finite}")
    checks.append(("6.X 무결성", ok6, "shape (N,1,28,20), no NaN/inf"))

    # ── 7. 재현성 기록 ────────────────────────────────────────────────────────
    L.append("\n## 7. seed/config 재현성 기록")
    L.append(f"- split_id (동결): `{SPLIT_SEED}`")
    L.append(f"- train seed (run config 고정 필요): **{TRAIN_SEED}**")
    L.append(f"- config: {CONFIG}")
    L.append("- 동일 cache + 동일 seed → 동일 결과 재현 보장 조건 기록 완료.")
    L.append("- ⚠ non-WALK sealed test N=27 경계선 → seed 민감도 가능. 가능하면 3~5 seed multi-seed, "
             "단일 seed면 seed 값 명시 + residual risk 기록.")
    checks.append(("7.재현성 기록", True, "seed/config recorded"))

    # ── 종합 ──────────────────────────────────────────────────────────────────
    all_ok = all(ok for _n, ok, _d in checks)
    L.append("\n## 종합 판정")
    L.append("check | 결과 | 비고")
    L.append("---|---|---")
    for n, ok, d in checks:
        L.append(f"{n} | {'PASS ✓' if ok else 'FAIL ✗'} | {d}")
    L.append(f"\n### {'✅ SANITY GATE PASS — GPU 학습 진행 가능' if all_ok else '❌ SANITY GATE FAIL — GPU 학습 금지, 위 FAIL 먼저 해결'}")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[생성] {OUT_MD}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
