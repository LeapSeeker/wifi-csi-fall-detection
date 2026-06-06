"""manifest_v2_manual_augmented 생성 (read-only).

진규 검수 결과(review_decisions.csv 98 + review_decisions_normal.csv 17)를 v1 manifest 와
병합해 v2 를 만든다. Codex 확정 스키마/케이스 매핑 반영.

핵심:
  - 케이스별 매핑(지시서 3): exclude 도 usable_for_fixed=true (fixed 엔 살림, 데이터 손실 방지).
  - exclude_reason_primary(지시서 4): 검수 4종 → Codex 5종. no_clear_transient/other 중
    high-noise 세션은 low_quality_env_subject 로 **세션 단위** 재분류(환경 일괄 분류 금지).
  - final onset median(지시서 5): 확정분만으로 산출. **median imputation 절대 금지**
    (제외 세션 onset=null, fallback 안 채움).
  - usable N(지시서 6): usable_for_onset_aligned=true 를 split/subtype/env/non-WALK pooled 집계.

제약: read-only. 원본·동결파일·v1 manifest 무수정. v2 신규 생성(finalization/).
"""
from __future__ import annotations
import csv
import json
import sys
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FINAL = ROOT / "debug/modeling/diag_out/onset_detector/finalization"
V1 = FINAL / "manifest_v1_auto_reviewed.csv"
RD_HIGH = FINAL / "review_tool/review_decisions.csv"
RD_NORM = FINAL / "review_tool_normal/review_decisions_normal.csv"
OUT_CSV = FINAL / "manifest_v2_manual_augmented.csv"
OUT_JSON = FINAL / "manifest_v2_summary.json"
OUT_MD = FINAL / "manifest_v2_summary.md"

# gate2 SOFT_HIGH — high-noise(환경 품질) 판정 임계 (세션 단위)
NOISE_HI = 1.206
MAD_HI = 0.079

SUBTYPES = ["FALL_WALK_F", "FALL_WALK_B", "FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B"]
WALK = {"FALL_WALK_F", "FALL_WALK_B"}
SPLITS = ["train", "val", "test", "out_of_scope"]


def fnum(x):
    try:
        if x in ("", "None", None):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def load_decisions():
    dec = {}
    for path, tag in ((RD_HIGH, "high"), (RD_NORM, "normal")):
        if not path.exists():
            print(f"[중단] 검수 결과 없음: {path}")
            sys.exit(1)
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            fn = r["filename"]
            if fn in dec:
                print(f"[경고] 중복 검수 결정: {fn} ({dec[fn]['_src']} & {tag})")
            r["_src"] = tag
            dec[fn] = r
    return dec


def main():
    v1_rows = list(csv.DictReader(open(V1, encoding="utf-8-sig")))
    man = {r["filename"]: r for r in v1_rows}
    dec = load_decisions()
    print(f"[입력] v1 {len(v1_rows)}행 | 검수 결정 {len(dec)}개 (high {sum(1 for d in dec.values() if d['_src']=='high')} "
          f"+ normal {sum(1 for d in dec.values() if d['_src']=='normal')})")

    # merge 무결성: 검수 결정 → v1 pending_manual 매핑 확인
    not_in_v1 = [fn for fn in dec if fn not in man]
    not_pending = [fn for fn in dec if fn in man and man[fn]["source"] != "pending_manual"]
    if not_in_v1:
        print(f"[중단] 검수 결정이 v1 에 없음: {not_in_v1}")
        return 1
    if not_pending:
        print(f"[경고] 검수했으나 v1 source!=pending_manual: {[(fn, man[fn]['source']) for fn in not_pending]}")
    dup = [fn for fn, c in Counter(list(dec)).items() if c > 1]
    print(f"  merge: 총 {len(dec)}개 (중복 {len(dup)}) | v1 미존재 {len(not_in_v1)} | pending_manual 외 {len(not_pending)}")

    def is_high_noise(fn):
        m = man.get(fn, {})
        bn, bmad = fnum(m.get("baseline_noise_ratio")), fnum(m.get("baseline_mad_ratio"))
        return (bn is not None and bn > NOISE_HI) or (bmad is not None and bmad > MAD_HI)

    def map_primary(fn, raw):
        """검수 reason → Codex 5종 primary + secondary 태그 (세션 단위)."""
        if raw == "walking_residual":
            return "walking_residual", ""
        if raw == "beep_misfire":
            return "beep_misfire", ""
        if raw in ("no_clear_transient", "other", ""):
            if is_high_noise(fn):
                m = man.get(fn, {})
                return "low_quality_env_subject", f"reviewer:{raw or 'none'};high_noise(n={m.get('baseline_noise_ratio')},mad={m.get('baseline_mad_ratio')})"
            return "no_clear_transient", (f"reviewer:{raw}" if raw == "other" else "")
        return "no_clear_transient", f"reviewer:{raw}"

    # ── v2 행 생성 ────────────────────────────────────────────────────────────
    V2_EXTRA = ["onset_status", "usable_for_fixed", "usable_for_onset_aligned", "exclude_scope",
                "exclude_reason_primary", "exclude_reason_secondary"]
    v2 = []
    for m in v1_rows:
        fn = m["filename"]
        row = dict(m)
        row["manifest_version"] = "v2_manual_augmented"
        row["manifest_id"] = "onset_manifest_v2_2026-06-06"
        prim = sec = ""
        if m["source"] == "excluded":  # resampled<550 / 손상
            status = "data_invalid"
            uf, uoa, scope = False, False, "all_policies"
            prim = "data_short_or_corrupt"
            on_o = on_c = ""
        elif fn in dec:
            d = dec[fn]
            if d["decision"] in ("approve", "modify"):
                status = "manual_corrected"
                uf, uoa, scope = True, True, "none"
                on_o = d.get("final_onset_frame_original", "") or d.get("recommended_onset", "")
                on_c = d.get("final_onset_frame_clean", "")
                if on_o == "":
                    print(f"[경고] {d['decision']} 인데 onset 없음: {fn} → pending 취급 권장")
            else:  # exclude
                status = "onset_unusable"
                uf, uoa, scope = True, False, "onset_aligned_only"  # 중요: fixed 엔 살림
                prim, sec = map_primary(fn, d.get("exclude_reason", ""))
                on_o = on_c = ""
        elif m["source"] == "auto_reviewed":
            status = "auto_reviewed"
            uf, uoa, scope = True, True, "none"
            on_o = m.get("onset_frame_original", "")
            on_c = m.get("onset_frame_clean", "")
        else:  # pending_manual 미검수 (test/out_of_scope 등)
            status = "pending_manual"
            uf, uoa, scope = True, False, "none"
            on_o = on_c = ""
        row["onset_frame_original"] = on_o
        row["onset_frame_clean"] = on_c
        row["onset_status"] = status
        row["usable_for_fixed"] = uf
        row["usable_for_onset_aligned"] = uoa
        row["exclude_scope"] = scope
        row["exclude_reason_primary"] = prim
        row["exclude_reason_secondary"] = sec
        v2.append(row)

    cols = list(v1_rows[0].keys()) + [c for c in V2_EXTRA if c not in v1_rows[0]]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(cols)
        for r in v2:
            w.writerow([r.get(c, "") for c in cols])
    print(f"[생성] {OUT_CSV} ({len(v2)}행)")

    # ── 집계 ─────────────────────────────────────────────────────────────────
    def es_of(m):
        return f"E{m.get('env','?')}_S{int(m.get('subject','0')):02d}" if m.get("subject") else "?"

    status_cnt = Counter(r["onset_status"] for r in v2)
    uf_n = sum(1 for r in v2 if r["usable_for_fixed"])
    uoa_rows = [r for r in v2 if r["usable_for_onset_aligned"]]
    uoa_n = len(uoa_rows)

    def by(rows, keyfn, keys=None):
        c = Counter(keyfn(r) for r in rows)
        return {k: c.get(k, 0) for k in (keys or sorted(c))}

    uoa_split = by(uoa_rows, lambda r: r["split_assignment"], SPLITS)
    uoa_sub = by(uoa_rows, lambda r: r["subtype"], SUBTYPES)
    uoa_env = by(uoa_rows, es_of)
    uf_split = by([r for r in v2 if r["usable_for_fixed"]], lambda r: r["split_assignment"], SPLITS)
    uf_sub = by([r for r in v2 if r["usable_for_fixed"]], lambda r: r["subtype"], SUBTYPES)

    # non-WALK pooled (train+val, onset-aligned 메인 비교 대상)
    nonwalk_tv = [r for r in uoa_rows if r["subtype"] not in WALK and r["split_assignment"] in ("train", "val")]
    walk_tv = [r for r in uoa_rows if r["subtype"] in WALK and r["split_assignment"] in ("train", "val")]

    # exclude_reason_primary 분포
    excl_rows = [r for r in v2 if r["exclude_reason_primary"]]
    prim_cnt = Counter(r["exclude_reason_primary"] for r in excl_rows)
    lowq = [r for r in v2 if r["exclude_reason_primary"] == "low_quality_env_subject"]

    # ── final onset median (확정분만, imputation 금지) ─────────────────────────
    def med_clean(rows):
        xs = [fnum(r["onset_frame_clean"]) for r in rows]
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 1) if xs else None

    manual = [r for r in uoa_rows if r["onset_status"] == "manual_corrected"]
    auto = [r for r in uoa_rows if r["onset_status"] == "auto_reviewed"]
    onset = {
        "overall_median_clean": med_clean(uoa_rows),
        "by_subtype": {st: med_clean([r for r in uoa_rows if r["subtype"] == st]) for st in SUBTYPES},
        "by_split": {sp: med_clean([r for r in uoa_rows if r["split_assignment"] == sp]) for sp in SPLITS},
        "manual_corrected_median": med_clean(manual),
        "auto_reviewed_median": med_clean(auto),
        "n_manual": len(manual), "n_auto": len(auto),
        "note": "imputation 없음 — 제외 세션 onset=null, median fallback 미사용. median 은 요약 통계용만.",
    }

    def tier(n):
        return "결론금지(<10)" if n < 10 else ("exploratory(10-19)" if n < 20 else "제한적해석(20+)")

    payload = {
        "meta": {"manifest_version": "v2_manual_augmented", "total": len(v2),
                 "inputs": {"v1": len(v1_rows), "reviewed_high": 98, "reviewed_normal": 17},
                 "low_quality_rule": f"per-session baseline_noise_ratio>{NOISE_HI} OR mad_ratio>{MAD_HI} (환경일괄 금지)"},
        "onset_status": dict(status_cnt),
        "usable_for_fixed": {"total": uf_n, "by_split": uf_split, "by_subtype": uf_sub},
        "usable_for_onset_aligned": {"total": uoa_n, "by_split": uoa_split, "by_subtype": uoa_sub, "by_env": uoa_env,
                                     "nonwalk_pooled_train_val": len(nonwalk_tv),
                                     "walk_pooled_train_val": len(walk_tv)},
        "exclude_reason_primary": dict(prim_cnt),
        "low_quality_sessions": [f"{es_of(man[r['filename']])} {r['subtype']} {r['filename']} "
                                 f"(n={man[r['filename']].get('baseline_noise_ratio')})" for r in lowq],
        "final_onset": onset,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── md 보고서 ─────────────────────────────────────────────────────────────
    L = ["# manifest_v2_manual_augmented 요약\n",
         f"총 {len(v2)} 세션 | v1 {len(v1_rows)} + 검수 115(high 98 + normal 17) 반영.",
         f"low_quality 판정: per-session noise>{NOISE_HI} OR mad>{MAD_HI} (환경 일괄 분류 금지).\n",
         "## onset_status 분포"]
    for k in ["auto_reviewed", "manual_corrected", "onset_unusable", "pending_manual", "data_invalid"]:
        L.append(f"- {k}: {status_cnt.get(k,0)}")
    L.append(f"\n## usable_for_fixed = {uf_n} / {len(v2)}")
    L.append("split: " + ", ".join(f"{k} {v}" for k, v in uf_split.items() if v))
    L.append(f"\n## usable_for_onset_aligned = {uoa_n} / {len(v2)}  ← 항목4 대상")
    L.append("split: " + ", ".join(f"{k} {v}" for k, v in uoa_split.items() if v))
    L.append("\n### subtype별 usable_for_onset_aligned (해석 tier)")
    L.append("subtype | n | tier")
    L.append("---|---|---")
    for st in SUBTYPES:
        L.append(f"{st} | {uoa_sub[st]} | {tier(uoa_sub[st])}")
    L.append("\n### env×subject별 usable_for_onset_aligned")
    for es in sorted(uoa_env):
        L.append(f"- {es}: {uoa_env[es]}")
    L.append(f"\n### ★ non-WALK pooled (train+val) usable_for_onset_aligned = {len(nonwalk_tv)} → {tier(len(nonwalk_tv))}")
    L.append(f"   WALK pooled (train+val) = {len(walk_tv)} → {tier(len(walk_tv))}")
    L.append(f"\n## exclude_reason_primary 분포 (제외 {len(excl_rows)})")
    for k in ["walking_residual", "beep_misfire", "no_clear_transient", "low_quality_env_subject", "data_short_or_corrupt"]:
        L.append(f"- {k}: {prim_cnt.get(k,0)}")
    L.append(f"\n### low_quality_env_subject 세션 ({len(lowq)}) — 세션단위 판정")
    for r in lowq:
        m = man[r["filename"]]
        L.append(f"- {es_of(m)} {r['subtype']} noise={m.get('baseline_noise_ratio')} {r['filename']}")
    L.append("\n## final onset median (clean, imputation 없음)")
    L.append(f"- overall: {onset['overall_median_clean']}  (manual {onset['manual_corrected_median']} / auto {onset['auto_reviewed_median']})")
    L.append(f"- by_subtype: {onset['by_subtype']}")
    L.append(f"- by_split: {onset['by_split']}")
    L.append("- ⚠ median imputation 금지: 제외 세션에 median onset 미부여(편향 유입 방지).")
    L.append("\n## 항목4 진행 가능성 소견")
    nwtier = tier(len(nonwalk_tv))
    L.append(f"- non-WALK pooled(train+val) {len(nonwalk_tv)}개 → **{nwtier}**. "
             + ("메인 paired 비교 가능." if len(nonwalk_tv) >= 20 else "메인 결론 무리, exploratory 한정."))
    L.append(f"- WALK 은 {len(walk_tv)}개 → {tier(len(walk_tv))}. subtype별로는 더 적어 별도 해석 주의.")
    L.append("- fixed 는 onset_unusable 포함 usable_for_fixed=true 라 데이터 손실 없음.")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[생성] {OUT_CSV}\n[생성] {OUT_JSON}\n[생성] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
