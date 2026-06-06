"""normal-priority(train/val) onset 검수 준비 도구 v2 (read-only).

진규의 priority-high 98개 검수가 끝난 뒤, manifest_v1_auto_reviewed.csv 에 남은
normal-priority train/val pending 17개를 추가 검수할 수 있도록 v2(인터랙티브 차트) 형식으로
준비한다. 기존 98개 v2 도구(review_tool/)와 **별도** 산출물로 만들어 동결물·판정을 안 섞는다.

생성물(모두 신규, finalization/ 하위):
  - review_tool_normal/energy_curves.json   17개 sparse-energy 곡선 (export_energy_curves.py 와
                                            1:1 동일 계산 = gate2 recompute_energy. 더미 금지.)
  - normal_review_queue.csv                 17행 (priority_review_queue.csv 와 동일 스키마)
  - review_tool_normal/onset_review.html    v2 인터랙티브 도구 (localStorage 키·Export 파일명 분리)
  - review_tool_normal/README.md

제약(read-only): 원본 데이터·manifest·동결 산출물(priority_review_queue.csv, review_tool/,
plots_priority/, pipeline/rpca 등) 무수정. 동결 코드는 import 만 한다.
계산식·상수는 export_energy_curves.py / gate2_onset_manifest_v1.py 와 동일(곡선 일치 보장).
17개 곡선은 실제 원본(data/cleaned/) 재계산 — 못 찾으면 즉시 중단.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # gate2 / build_onset_review_tool / export_energy_curves

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import export_energy_curves as ec            # noqa: E402  곡선 계산(= gate2 recompute_energy) 재사용
import gate2_onset_manifest_v1 as g2          # noqa: E402  candidates_topk / 상수 / recommend 로직
import build_onset_review_tool as bt          # noqa: E402  v2 HTML 템플릿 재사용

FINAL = ROOT / "debug/modeling/diag_out/onset_detector/finalization"
MANIFEST_V1 = FINAL / "manifest_v1_auto_reviewed.csv"
NORMAL_QUEUE = FINAL / "normal_review_queue.csv"
TOOL_DIR = FINAL / "review_tool_normal"
CURVES_JSON = TOOL_DIR / "energy_curves.json"

QUEUE_COLS = ["rank", "filename", "subtype", "split_assignment", "review_priority", "hard_reasons",
              "soft_reasons", "soft_count", "baseline_noise_ratio", "param_sensitivity", "topk_spread",
              "peak_over_baseline", "topk_count", "walk_baseline_contamination",
              "topk_cand_frames_clean", "rise_frame_original", "exclude_candidate", "recommendation"]


def select_17(rows):
    return [r for r in rows
            if r["review_priority"] == "normal"
            and r["needs_review"] == "True"
            and r["split_assignment"] in ("train", "val")]


def soft_to_comma(soft_reasons):
    return ",".join([x for x in (soft_reasons or "").split(";") if x])


def recommend(r, pob, topk, exclude_candidate, rise_not_found):
    note = g2.RECO_NOTE
    if exclude_candidate:
        return f"exclude candidate (no_clear_transient): search max {pob:.2f}x baseline, 0 sustained candidate. {note}"
    if rise_not_found:
        if pob is not None and pob >= g2.PEAK_OVER_BASELINE_FLOOR:
            return f"weak transient (search max {pob:.2f}x baseline, no sustained-5). manual confirm onset. {note}"
        return f"no sustained crossing; manual confirm. {note}"
    rise_o = g2.fnum(r["rise_frame_original"])
    rise_c = r["rise_frame_clean"]
    hard = r["needs_review_hard_reasons"]
    in_win = [c for c in topk if c["in_fall_window"] and not c["in_beep"]]
    if hard and ("beep" in hard or "too_early" in hard):
        if in_win:
            c = in_win[0]
            return (f"auto rise(frame {rise_o}) in beep/early; strongest in-window candidate "
                    f"frame {c['frame_original']}(clean {c['frame_clean']}). manual confirm. {note}")
        return (f"auto rise(frame {rise_o}) in beep/early; no clean in-window candidate "
                f"→ likely no_clear_transient. manual review. {note}")
    if rise_c != "":
        return (f"auto rise frame {rise_o}(clean {rise_c}) but weak "
                f"(soft: {soft_to_comma(r['needs_review_soft_reasons'])}). likely OK, manual confirm. {note}")
    return f"manual confirm. {note}"


def sort_key(row):
    split_rank = 0 if row["split_assignment"] in ("train", "val") else 1
    prio_rank = 0 if row["review_priority"] == "high" else 1
    hard_rank = 0 if row["needs_review_hard_reasons"] else 1
    bn = -(g2.fnum(row["baseline_noise_ratio"]) or 0)
    ps = -(g2.fnum(row["param_sensitivity"]) or 0)
    st = row["subtype"]
    sub_rank = 0 if st == "FALL_WALK_B" else (1 if st == "FALL_WALK_F" else 2)
    return (split_rank, prio_rank, hard_rank, bn, ps, sub_rank)


def main():
    if not MANIFEST_V1.exists():
        print(f"[중단] manifest 없음: {MANIFEST_V1}")
        return 1
    rows = list(csv.DictReader(open(MANIFEST_V1, encoding="utf-8-sig")))
    sel = sorted(select_17(rows), key=sort_key)
    print(f"[추출] normal-priority train/val pending = {len(sel)}개")
    if len(sel) != 17:
        print(f"[경고] 기대 17개와 다름({len(sel)}). 계속 진행하되 확인 필요.")

    TOOL_DIR.mkdir(parents=True, exist_ok=True)
    bstats = ec.load_baseline_stats()  # export_energy_curves 와 동일 출처(MANIFEST_V0 base 행)

    curves = {}
    queue_rows = []
    aborted = []
    t0 = time.time()
    for i, r in enumerate(sel):
        fn = r["filename"]
        es = ec.recompute_energy(fn)  # 실제 원본 재계산 — 더미 아님
        bm, bmad = bstats.get(fn, (None, None))
        if es is None or bm is None or bmad is None:
            aborted.append(fn)
            print(f"  [{i+1}/{len(sel)}] {fn}  곡선/baseline 없음 → 중단 대상")
            continue
        thr = bm + ec.K_FIX * bmad
        curves[fn] = {
            "e": [round(float(v), 5) for v in es],
            "thr": round(float(thr), 5),
            "bm": round(float(bm), 5),
            "bmad": round(float(bmad), 5),
            "ymax": round(float(es.max()), 5),
        }
        pob = float(es[ec.SEARCH[0]:ec.SEARCH[1]].max() / (bm + 1e-9))
        topk, ncand, _ = g2.candidates_topk(es, bm, bmad)
        hard = r["needs_review_hard_reasons"] or ""
        rise_not_found = "rise_not_found" in hard
        exclude_candidate = rise_not_found and pob < g2.PEAK_OVER_BASELINE_FLOOR and ncand == 0
        reco = recommend(r, pob, topk, exclude_candidate, rise_not_found)
        walkcontam = (r["subtype"] in g2.WALK_SUBTYPES
                      and (g2.fnum(r["baseline_noise_ratio"]) or 0) > g2.SOFT_HIGH["baseline_noise_ratio"])
        queue_rows.append({
            "rank": 0, "filename": fn, "subtype": r["subtype"], "split_assignment": r["split_assignment"],
            "review_priority": r["review_priority"], "hard_reasons": hard,
            "soft_reasons": soft_to_comma(r["needs_review_soft_reasons"]),
            "soft_count": r["soft_warning_count"],
            "baseline_noise_ratio": r["baseline_noise_ratio"], "param_sensitivity": r["param_sensitivity"],
            "topk_spread": r["topk_spread"],
            "peak_over_baseline": round(pob, 3), "topk_count": ncand,
            "walk_baseline_contamination": walkcontam,
            "topk_cand_frames_clean": "|".join(str(c["frame_clean"]) for c in topk),
            "rise_frame_original": r["rise_frame_original"],
            "exclude_candidate": exclude_candidate, "recommendation": reco,
        })
        print(f"  [{i+1}/{len(sel)}] {fn}  es_max={es.max():.4f} thr={thr:.4f} pob={pob:.2f} ncand={ncand}")

    if aborted:
        print(f"[중단] 원본/baseline 없음 {len(aborted)}개 — 더미 생성 안 함: {aborted}")
        return 2

    for i, qr in enumerate(queue_rows, 1):
        qr["rank"] = i

    # ── energy_curves.json (v2 차트용, export_energy_curves 와 동일 스키마) ──────
    payload = {
        "meta": {
            "source": "build_normal_review_prep.py (= export_energy_curves / gate2 recompute_energy)",
            "set": "normal_priority_train_val_17",
            "n_frames": 550,
            "smooth_frames": ec.SMOOTH, "k_mad": ec.K_FIX, "sustain_frames": ec.SUSTAIN_FIX,
            "baseline": list(ec.BASELINE), "search": list(ec.SEARCH),
            "fall_window": list(ec.FALL_ORIG), "beep_regions": [list(b) for b in ec.BEEP_REGIONS],
            "base_param_set": ec.BASE_PARAM_SET,
        },
        "curves": curves,
    }
    CURVES_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[생성] {CURVES_JSON}  곡선 {len(curves)}/{len(sel)} | {CURVES_JSON.stat().st_size/1024:.0f}KB")

    # ── normal_review_queue.csv ───────────────────────────────────────────────
    with open(NORMAL_QUEUE, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(QUEUE_COLS)
        for qr in queue_rows:
            w.writerow([qr[c] for c in QUEUE_COLS])
    print(f"[생성] {NORMAL_QUEUE} ({len(queue_rows)}행)")

    # ── v2 인터랙티브 검수 도구 (기존 템플릿 재사용, normal 전용 분리) ──────────
    bt.QUEUE = NORMAL_QUEUE
    sessions = bt.build_sessions()
    curves_text = CURVES_JSON.read_text(encoding="utf-8")
    html = (bt.HTML
            .replace("__DATA__", json.dumps(sessions, ensure_ascii=False))
            .replace("__CURVES__", curves_text)
            .replace("../plots_priority/", "../plots_normal/")  # PNG 폴백 경로(전 곡선 임베드라 미발생)
            .replace('"onset_review_decisions_v1"', '"onset_review_decisions_normal_v1"')
            .replace("review_decisions.csv", "review_decisions_normal.csv")
            .replace("review_decisions.json", "review_decisions_normal.json")
            .replace("SafeSignal onset 검수 v2", "SafeSignal onset 검수 v2 (normal 17)"))
    (TOOL_DIR / "onset_review.html").write_text(html, encoding="utf-8")
    matched = sum(1 for s in sessions if s["filename"] in curves)
    readme = (
        "# onset 검수 도구 (normal-priority 17개, v2 인터랙티브 차트)\n\n"
        "priority-high 98개 검수 후, manifest_v1 의 normal-priority train/val pending 17개 추가 검수용.\n"
        "기존 98개 도구(review_tool/)와 **별도** — localStorage 키(onset_review_decisions_normal_v1)와\n"
        "Export 파일명(review_decisions_normal.csv)이 분리되어 98개 판정과 섞이지 않음.\n\n"
        "1. `onset_review.html` 을 브라우저로 더블클릭(곡선 임베드라 file:// 에서도 표시).\n"
        "2. 차트: 파란 굵은 선=sparse energy(판단 기준). 구간 빗금/글자라벨, x눈금 1·10·100 차등,\n"
        "   hover frame·energy, 클릭=선택 후보(초록선) → M 으로 수정 저장. (98개 도구와 동일 형식)\n"
        "3. A=승인 / C=제외 / M=수정 / ←→=이동. 제외 사유 드롭다운은 4종 —\n"
        "   Codex 5-primary(walking_residual/beep_misfire/no_clear_transient/low_quality_env_subject/\n"
        "   data_short_or_corrupt) 재분류는 export 후 baseline_noise_ratio 교차참조로 별도 수행.\n"
        "4. 끝나면 Export CSV → review_decisions_normal.csv → finalization/review_tool_normal/ 에 두고\n"
        "   Claude Code 에 v2 manifest 생성 요청(98 + 17 통합).\n\n"
        "판정은 진규. 추천은 참고용. 원본/동결 파일 무수정.\n"
    )
    (TOOL_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"[생성] {TOOL_DIR}/onset_review.html (곡선 매칭 {matched}/{len(sessions)}), README.md")

    # ── 분포 요약 ─────────────────────────────────────────────────────────────
    from collections import Counter
    sub = Counter(qr["subtype"] for qr in queue_rows)
    env = Counter(f"E{r['env']}_S{int(r['subject']):02d}" for r in sel)
    walk = sum(v for k, v in sub.items() if "WALK" in k)
    print(f"\n=== 17개 분포 ({time.time()-t0:.0f}s) ===")
    print(f"  subtype: {dict(sub)}")
    print(f"  WALK 비율: {walk}/{len(queue_rows)} ({100*walk/len(queue_rows):.1f}%)")
    print(f"  env×subject: {dict(env)}")
    print(f"  exclude_candidate(자동 제외 후보): {sum(1 for qr in queue_rows if qr['exclude_candidate'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
