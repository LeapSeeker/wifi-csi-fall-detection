"""항목4 선행: fixed ∩ onset_primary paired N 집계 (read-only).

Gate 3 crop_index.csv 만 사용. paired = 같은 세션에서 fixed 생성 AND onset_primary 실제 생성
(crop_out_of_bounds 아닌). split/subtype/WALK 분해 + 판정 적용 후 결론까지 보고.
read-only: crop_index.csv 외 입력 없음. 산출 없음(보고만, md 저장).
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CI = ROOT / "debug/modeling/diag_out/onset_detector/gate3_cache/crop_index.csv"
OUT_MD = ROOT / "debug/modeling/diag_out/onset_detector/gate3_cache/paired_n_report.md"
WALK = {"FALL_WALK_F", "FALL_WALK_B"}
NONWALK_SUB = ["FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B"]
SPLITS = ["train", "val", "test"]


def tier(n):
    if n < 10:
        return "exploratory(정식 결론 금지)"
    if n < 20:
        return "제한적 결론(CI 중심)"
    return "main 비교 가능(20+)"


def main():
    rows = list(csv.DictReader(open(CI, encoding="utf-8-sig")))
    # filename -> {policy: generated_bool}, + meta
    sess = defaultdict(dict)
    meta = {}
    for r in rows:
        fn = r["filename"]
        sess[fn][r["crop_policy"]] = (r["generated"] == "True")
        meta[fn] = {"subtype": r["subtype"], "split": r["split_assignment"]}

    def gen(fn, p):
        return sess[fn].get(p, False)

    paired_primary = [fn for fn in sess if gen(fn, "fixed") and gen(fn, "onset_primary")]
    paired_reduced = [fn for fn in sess if gen(fn, "fixed") and gen(fn, "onset_reduced")]

    def split_cnt(fns, filt=None):
        c = Counter(meta[fn]["split"] for fn in fns if (filt is None or filt(meta[fn])))
        return {s: c.get(s, 0) for s in SPLITS}

    def is_nonwalk(m): return m["subtype"] not in WALK
    def is_walk(m): return m["subtype"] in WALK

    p_all = split_cnt(paired_primary)
    p_nonwalk = split_cnt(paired_primary, is_nonwalk)
    p_walk = split_cnt(paired_primary, is_walk)
    p_bysub = {st: split_cnt(paired_primary, lambda m, st=st: m["subtype"] == st) for st in NONWALK_SUB}

    # OOB (onset_primary)
    oob_rows = [r for r in rows if r["crop_policy"] == "onset_primary" and r["crop_exclude_reason"] == "crop_out_of_bounds"]
    oob_split = {s: sum(1 for r in oob_rows if r["split_assignment"] == s) for s in SPLITS}
    oob_sub = dict(Counter(r["subtype"] for r in oob_rows))

    r_all = split_cnt(paired_reduced)
    r_nonwalk = split_cnt(paired_reduced, is_nonwalk)

    def row(label, d):
        return f"{label} | {d['train']} | {d['val']} | {d['test']} | {d['train']+d['val']} | {sum(d.values())}"

    L = ["# 항목4 선행 — fixed ∩ onset_primary paired N 집계\n",
         "paired = 동일 세션에서 fixed 생성 AND onset_primary 실제 생성(crop_out_of_bounds 제외). crop_index.csv 기준.\n",
         "## 1) paired 전체 / 2) non-WALK / 3) WALK (split별)",
         "구분 | train | val | test | train+val | 합계",
         "---|---|---|---|---|---",
         row("전체 paired", p_all),
         row("non-WALK paired", p_nonwalk),
         row("WALK paired", p_walk),
         "",
         "## 4) non-WALK subtype별 paired (split별)",
         "subtype | train | val | test | train+val | 합계",
         "---|---|---|---|---|---"]
    for st in NONWALK_SUB:
        L.append(row(st.replace("FALL_", ""), p_bysub[st]))
    L.append("")
    L.append("## 5) onset_primary crop_out_of_bounds")
    L.append(f"- split별: train {oob_split['train']} / val {oob_split['val']} / test {oob_split['test']} (합 {sum(oob_split.values())})")
    L.append("- subtype별: " + ", ".join(f"{k.replace('FALL_','')} {v}" for k, v in sorted(oob_sub.items())))
    L.append("")
    L.append("## 6) 참고 — fixed ∩ onset_reduced paired (ablation)")
    L.append("구분 | train | val | test | train+val | 합계")
    L.append("---|---|---|---|---|---")
    L.append(row("전체 paired(reduced)", r_all))
    L.append(row("non-WALK paired(reduced)", r_nonwalk))
    L.append("")
    L.append("## 판정 (non-WALK val/test 기준)")
    nv, nt = p_nonwalk["val"], p_nonwalk["test"]
    L.append(f"- non-WALK **val N={nv}** → {tier(nv)}")
    L.append(f"- non-WALK **test N={nt}** → {tier(nt)}")
    L.append(f"- non-WALK train N={p_nonwalk['train']} (학습 coverage 근거, 성능 N 아님)")
    L.append(f"- train+val={p_nonwalk['train']+p_nonwalk['val']} 은 개발 coverage 근거로만(성능 결론 N 아님).")
    L.append("")
    L.append("## reduced vs primary paired N")
    L.append(f"- non-WALK paired: primary train+val {p_nonwalk['train']+p_nonwalk['val']} vs reduced {r_nonwalk['train']+r_nonwalk['val']}"
             f" → reduced가 {'더 많음' if (r_nonwalk['train']+r_nonwalk['val'])>(p_nonwalk['train']+p_nonwalk['val']) else '더 적거나 같음'} (OOB 차이 반영)")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[생성] {OUT_MD}")


if __name__ == "__main__":
    main()
