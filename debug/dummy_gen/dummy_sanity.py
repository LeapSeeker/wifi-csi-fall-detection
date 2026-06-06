"""더미 생성 1차 — Sanity + clean400 detector calibration (read-only).

1) split 고정 확인: train non-WALK fall(auto/manual 확정 onset)만 origin. val/test 미포함.
2) origin 분포(subtype/subject/env) 보고.
3) calibration: origin clean400(증강 없음)에 clean400 detector 재검출 → manifest onset_clean 과 비교.
   |detected - manifest| 분포가 작아야 onset 교차검증(expected vs detected) 유효.
"""
from __future__ import annotations
import csv
import sys
from collections import Counter
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
CALIB_N = 30  # calibration 표본(전체 RPCA는 느림)


def select_origins():
    rows = list(csv.DictReader(open(V2, encoding="utf-8-sig")))
    sel = []
    for r in rows:
        if r["split_assignment"] != "train":
            continue
        if r["subtype"] in WALK:
            continue
        if r["onset_status"] not in ("auto_reviewed", "manual_corrected"):
            continue
        if r["onset_frame_clean"] in ("", "None"):
            continue
        sel.append(r)
    return sel, rows


def main():
    sel, rows = select_origins()
    print(f"[origin] train non-WALK fall (auto/manual 확정 onset) = {len(sel)}개")
    # split 누수 검사: 선택된 게 전부 train 인지
    bad = [r["filename"] for r in sel if r["split_assignment"] != "train"]
    print(f"  split=train 확인: 위반 {len(bad)} {'OK' if not bad else bad[:3]}")
    # 분포
    print(f"  subtype: {dict(Counter(r['subtype'] for r in sel))}")
    print(f"  onset_status: {dict(Counter(r['onset_status'] for r in sel))}")
    print(f"  env: {dict(Counter(r['env'] for r in sel))} | subject: {dict(Counter(r['subject'] for r in sel))}")
    print(f"  ×3 생성 예정: {len(sel) * 3}개")
    # 참고: 제외될 origin (onset_unusable/no_clear_transient/data_invalid) train non-WALK
    excl = [r for r in rows if r["split_assignment"] == "train" and r["subtype"] not in WALK
            and r["onset_status"] in ("onset_unusable", "pending_manual", "data_invalid")]
    print(f"  (참고) train non-WALK 중 origin 제외(onset 없음): {len(excl)}")

    # ── calibration: clean400 detector vs manifest onset ──────────────────────
    print(f"\n[calibration] clean400 detector vs manifest onset_clean (표본 {CALIB_N})")
    import random
    random.seed(42)
    samp = random.sample(sel, min(CALIB_N, len(sel)))
    diffs, rnf, miss = [], 0, 0
    by_status = {"auto_reviewed": [], "manual_corrected": []}
    for r in samp:
        p = next(CLEANED.rglob(r["filename"]), None)
        if p is None:
            miss += 1
            continue
        c4 = L.build_clean400(p)
        if c4 is None:
            miss += 1
            continue
        det = L.detect_onset_clean(c4)
        man_on = int(float(r["onset_frame_clean"]))
        if det["rise"] is None:
            rnf += 1
            continue
        d = det["rise"] - man_on
        diffs.append(d)
        by_status[r["onset_status"]].append(abs(d))
    if diffs:
        ad = np.abs(diffs)
        print(f"  표본 {len(diffs)} | |diff| median={np.median(ad):.1f} p90={np.percentile(ad,90):.1f} "
              f"max={ad.max()} | signed mean={np.mean(diffs):+.1f}")
        print(f"  |diff|<=5: {int((ad<=5).sum())}/{len(ad)} | <=10: {int((ad<=10).sum())}/{len(ad)}")
        for st, v in by_status.items():
            if v:
                print(f"    {st}: |diff| median={np.median(v):.1f} (n={len(v)})")
    print(f"  rise_not_found {rnf} | data/경로 누락 {miss}")
    print("\n[판정] |diff| median<=3 이고 <=5 비율 높으면 detector calibration OK → 생성 진행 가능.")
    print("       auto는 작아야 정상(detector 동일계열). manual은 검수자 보정이라 다소 클 수 있음(pending 후보).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
