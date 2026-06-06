"""onset 검수 exclude 57개 → Codex 5-primary reason 재분류·집계 (read-only).

review_tool/review_decisions.csv 의 exclude 57건을 검수도구 4종 사유에서 Codex 의
5-primary(상호배타) 체계로 재분류해 집계한다. 핵심(Codex #4): no_clear_transient 로 제외한
것 중 실제 환경/피험자 품질 문제(baseline 시끄러움)는 low_quality_env_subject 로 분리.
판정 근거는 manifest_v1 의 baseline_noise_ratio / baseline_mad_ratio (gate2 SOFT_HIGH 기준).

Codex 5-primary:
  walking_residual / beep_misfire / no_clear_transient / low_quality_env_subject / data_short_or_corrupt

read-only: 원본·manifest·review_decisions·동결물 무수정. 산출: finalization/exclude_reason_analysis.md
"""
from __future__ import annotations
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "debug/modeling/diag_out/onset_detector/finalization"
RD = FINAL / "review_tool/review_decisions.csv"
MAN = FINAL / "manifest_v1_auto_reviewed.csv"
OUT_MD = FINAL / "exclude_reason_analysis.md"

# gate2 SOFT_HIGH (high-noise) 임계 — 환경 품질 판정 기준
NOISE_HI = 1.206   # baseline_noise_ratio (= 전체 p90)
MAD_HI = 0.079     # baseline_mad_ratio

SUBTYPES = ["FALL_WALK_F", "FALL_WALK_B", "FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B"]
ENVSUBS = [f"E{e}_S{e if False else s:02d}" for e in (1, 2, 3, 4) for s in (1, 2, 3)]  # placeholder, rebuilt below


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    rd = list(csv.DictReader(open(RD, encoding="utf-8-sig")))
    man = {r["filename"]: r for r in csv.DictReader(open(MAN, encoding="utf-8-sig"))}
    exc = [r for r in rd if r["decision"] == "exclude"]

    def es_of(fn):
        m = man.get(fn, {})
        return f"E{m.get('env','?')}_S{int(m.get('subject','0')):02d}" if m.get("subject") else "?"

    def is_low_quality(fn):
        m = man.get(fn, {})
        bn = fnum(m.get("baseline_noise_ratio"))
        bmad = fnum(m.get("baseline_mad_ratio"))
        return (bn is not None and bn > NOISE_HI) or (bmad is not None and bmad > MAD_HI)

    # ── 재분류 ────────────────────────────────────────────────────────────────
    primary = {}   # fn -> primary reason
    for r in exc:
        fn = r["filename"]
        raw = r["exclude_reason"]
        m = man.get(fn, {})
        short = (m.get("source") == "excluded") or ("550" in (m.get("needs_review_hard_reasons") or ""))
        if short:
            primary[fn] = "data_short_or_corrupt"
        elif raw == "walking_residual":
            primary[fn] = "walking_residual"
        elif raw == "beep_misfire":
            primary[fn] = "beep_misfire"
        elif raw == "no_clear_transient":
            primary[fn] = "low_quality_env_subject" if is_low_quality(fn) else "no_clear_transient"
        else:  # 'other' 등
            primary[fn] = "low_quality_env_subject" if is_low_quality(fn) else "no_clear_transient"

    PR_ORDER = ["walking_residual", "beep_misfire", "no_clear_transient",
                "low_quality_env_subject", "data_short_or_corrupt"]
    pr_cnt = Counter(primary.values())

    # subtype/env 분포 (전체 57)
    sub_cnt = Counter(r["subtype"] for r in exc)
    env_cnt = Counter(es_of(r["filename"]) for r in exc)
    # primary × subtype, primary × env
    pr_sub = defaultdict(Counter)
    pr_env = defaultdict(Counter)
    for r in exc:
        fn = r["filename"]
        pr_sub[primary[fn]][r["subtype"]] += 1
        pr_env[primary[fn]][es_of(fn)] += 1

    # low_quality 상세 + 대안(Codex env 기반)
    lowq = [r["filename"] for r in exc if primary[r["filename"]] == "low_quality_env_subject"]
    nct = [r["filename"] for r in exc if primary[r["filename"]] == "no_clear_transient"]
    alt_codex_env = {"E1_S01", "E4_S01"}
    alt_lowq = [r["filename"] for r in exc
                if r["exclude_reason"] in ("no_clear_transient", "other") and es_of(r["filename"]) in alt_codex_env]

    L = []
    L.append("# onset exclude 57개 — Codex 5-primary 재분류 집계\n")
    L.append("입력: review_tool/review_decisions.csv (exclude 57), manifest_v1 baseline 지표.")
    L.append(f"low_quality 판정: baseline_noise_ratio>{NOISE_HI} OR baseline_mad_ratio>{MAD_HI} (gate2 SOFT_HIGH).\n")

    L.append("## 1) primary reason별 (상호배타)")
    L.append("reason | n")
    L.append("---|---")
    for k in PR_ORDER:
        L.append(f"{k} | {pr_cnt.get(k,0)}")
    L.append(f"**합계** | **{sum(pr_cnt.values())}**\n")

    L.append("## 2) subtype별 제외 (전체 57)")
    L.append("subtype | n")
    L.append("---|---")
    for st in SUBTYPES:
        L.append(f"{st} | {sub_cnt.get(st,0)}")
    L.append("")
    L.append("### primary × subtype")
    L.append("primary \\ subtype | " + " | ".join(s.replace("FALL_", "") for s in SUBTYPES))
    L.append("---|" + "|".join(["---"] * len(SUBTYPES)))
    for k in PR_ORDER:
        if pr_cnt.get(k, 0):
            L.append(f"{k} | " + " | ".join(str(pr_sub[k].get(st, 0)) for st in SUBTYPES))
    L.append("")

    L.append("## 3) env×subject별 제외 (전체 57)")
    envs = sorted(env_cnt)
    L.append("env_subject | n")
    L.append("---|---")
    for es in envs:
        L.append(f"{es} | {env_cnt[es]}")
    L.append("")
    L.append("### primary × env_subject")
    L.append("primary \\ env | " + " | ".join(envs))
    L.append("---|" + "|".join(["---"] * len(envs)))
    for k in PR_ORDER:
        if pr_cnt.get(k, 0):
            L.append(f"{k} | " + " | ".join(str(pr_env[k].get(es, 0)) for es in envs))
    L.append("")

    L.append("## 4) low_quality_env_subject 상세 (Codex #4)")
    L.append(f"노이즈 기준 적용 결과: **{len(lowq)}개**")
    for fn in lowq:
        m = man.get(fn, {})
        L.append(f"- {es_of(fn)} {man.get(fn,{}).get('subtype','')} "
                 f"noise={m.get('baseline_noise_ratio')} mad={m.get('baseline_mad_ratio')}  {fn}")
    L.append("")
    L.append("### ⚠ Codex 가정과의 불일치 (중요)")
    L.append("Codex 는 E4_S01·E1_S01 을 baseline 품질 나쁜 환경으로 지목했으나, manifest "
             "baseline_noise_ratio 상으로는 두 환경이 noisy 하지 않다:")
    nctall = [r["filename"] for r in exc if r["exclude_reason"] in ("no_clear_transient", "other")]
    by_es = defaultdict(list)
    for fn in nctall:
        by_es[es_of(fn)].append(fnum(man.get(fn, {}).get("baseline_noise_ratio")))
    L.append("env_subject | no_clear_transient n | mean noise | max noise")
    L.append("---|---|---|---")
    for es in sorted(by_es):
        vals = [v for v in by_es[es] if v is not None]
        L.append(f"{es} | {len(by_es[es])} | {sum(vals)/len(vals):.3f} | {max(vals):.3f}")
    L.append("")
    L.append("- 전체 세션 baseline_noise_ratio: median 1.101 / p90 1.202 / max 3.057.")
    L.append("- E1_S01 mean 1.115, E4_S01 mean 1.100 — **둘 다 거의 median 수준(=안 시끄러움)**.")
    L.append("- 실제 high-noise no_clear_transient 는 E2_S02(WALK_F 1.559) / E4_S03(SIT_B 1.206) 쪽.")
    L.append("")
    L.append(f"### 대안: Codex env 기반(E1_S01+E4_S01 의 NCT 전부 low_quality 로) → {len(alt_lowq)}개")
    L.append("  " + ", ".join(es_of(fn) + ":" + fn for fn in alt_lowq))
    L.append("> 권고: baseline_noise_ratio 가 E1_S01/E4_S01 의 품질 문제를 뒷받침하지 않으므로, "
             "low_quality 분리는 **노이즈 기준(상기 4번)**을 default 로 하고, E1_S01/E4_S01 의 "
             "'출렁임'(저주파 drift)은 noise_ratio 로 안 잡힐 수 있으니 해당 곡선 육안 확인 후 확정 권장.")
    L.append("")
    L.append("## 참고: data_short_or_corrupt")
    L.append("manual exclude 57 중 resampled<550/손상 = 0건 (단축/손상은 v1 auto-exclude 단계에서 이미 제외됨).")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[생성] {OUT_MD}")


if __name__ == "__main__":
    main()
