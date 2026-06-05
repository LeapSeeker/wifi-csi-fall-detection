"""onset 검수 도구용 sparse-energy 곡선 export (read-only, 1회성 파생물 생성기).

검수 도구 v2(인터랙티브 차트)가 hover/실시간 세로선/눈금 차등을 그리려면 프레임별
sparse-energy 곡선 데이터가 필요하다. 이 곡선은 gate2_onset_manifest_v1.py 가 실행 중
메모리(energy_cache)에서만 들고 PNG 로 렌더한 뒤 버려져 디스크에 없다.

이 스크립트는 priority_review_queue.csv 의 train/val 98 세션에 대해 PNG 와 *동일한*
계산(load → resample_to_100hz → rpca_sparse → mean|·| → 5-frame smoothing)을 재수행해
곡선 배열 + thr/baseline 메타를 finalization/review_tool/energy_curves.json 으로 내보낸다.

제약(read-only): 원본 데이터·plot png·동결 파일(pipeline/rpca/...)·manifest 무수정.
  이 스크립트는 *새 파생 산출물*(energy_curves.json)만 만든다. 동결 코드는 import 만 한다.
  계산식·상수는 gate2_onset_manifest_v1.py 와 1:1 동일해야 PNG 와 곡선이 일치한다.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from model.preprocessing.loader import load_safesignal_csv          # noqa: E402
from model.preprocessing.resample import resample_to_100hz          # noqa: E402
from model.preprocessing.rpca import rpca_sparse, DEFAULT_MAX_ITER  # noqa: E402

# ── gate2_onset_manifest_v1.py 와 동일 상수 (동결) ──────────────────────────────
CLEANED = ROOT / "data/cleaned"
OUT = ROOT / "debug/modeling/diag_out/onset_detector"
FINAL = OUT / "finalization"
QUEUE = FINAL / "priority_review_queue.csv"
MANIFEST_V0 = OUT / "onset_probe_manifest_long.csv"
TOOL_DIR = FINAL / "review_tool"
OUT_JSON = TOOL_DIR / "energy_curves.json"

BASELINE = (50, 150)
SEARCH = (190, 350)
SMOOTH = 5
K_FIX = 3.0
SUSTAIN_FIX = 5
FALL_ORIG = (200, 400)
BEEP_REGIONS = [(0, 50), (150, 200), (400, 450)]
BASE_PARAM_SET = "k3.0_s5"   # nominal / k_mad=3.0 / sustain=5 (확정 base detector)


def fnum(x):
    try:
        if x in ("", "None", None):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def recompute_energy(fname):
    """gate2_onset_manifest_v1.recompute_energy 와 동일 계산. 반환: es(550,) 또는 None."""
    path = next(CLEANED.rglob(fname), None)
    if path is None:
        return None
    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    if int(res.resampled_count) < 550:
        return None
    A = res.amplitude[:550]
    e = np.abs(rpca_sparse(A, max_iter=DEFAULT_MAX_ITER, tol=None)).mean(axis=1)
    h = SMOOTH // 2
    es = np.array([e[max(0, i - h):min(len(e), i + h + 1)].mean() for i in range(len(e))])
    return es


def load_baseline_stats():
    """long manifest base 행에서 filename → (baseline_median, baseline_mad). thr 계산용."""
    stats = {}
    with open(MANIFEST_V0, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("param_set_id") != BASE_PARAM_SET:
                continue
            if r.get("search_mode") != "nominal":
                continue
            fn = r["filename"]
            if fn in stats:
                continue
            stats[fn] = (fnum(r.get("baseline_median")), fnum(r.get("baseline_mad")))
    return stats


def main():
    TOOL_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(QUEUE, encoding="utf-8-sig")))
    rows = [r for r in rows if r["split_assignment"] in ("train", "val")]
    bstats = load_baseline_stats()
    print(f"[export] train/val {len(rows)} 세션 곡선 재계산 시작 (세션당 ~4s)")

    curves = {}
    t_all = time.time()
    miss = []
    for i, r in enumerate(rows):
        fn = r["filename"]
        t0 = time.time()
        es = recompute_energy(fn)
        if es is None:
            miss.append(fn)
            print(f"  [{i+1}/{len(rows)}] {fn}  MISSING/short → skip")
            continue
        bm, bmad = bstats.get(fn, (None, None))
        thr = (bm + K_FIX * bmad) if (bm is not None and bmad is not None) else None
        curves[fn] = {
            "e": [round(float(v), 5) for v in es],
            "thr": (round(float(thr), 5) if thr is not None else None),
            "bm": (round(float(bm), 5) if bm is not None else None),
            "bmad": (round(float(bmad), 5) if bmad is not None else None),
            "ymax": round(float(es.max()), 5),
        }
        print(f"  [{i+1}/{len(rows)}] {fn}  es_max={es.max():.4f} thr={thr} ({time.time()-t0:.1f}s)")

    payload = {
        "meta": {
            "source": "export_energy_curves.py (= gate2_onset_manifest_v1 recompute_energy)",
            "n_frames": 550,
            "smooth_frames": SMOOTH,
            "k_mad": K_FIX,
            "sustain_frames": SUSTAIN_FIX,
            "baseline": list(BASELINE),
            "search": list(SEARCH),
            "fall_window": list(FALL_ORIG),
            "beep_regions": [list(b) for b in BEEP_REGIONS],
            "base_param_set": BASE_PARAM_SET,
        },
        "curves": curves,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    sz = OUT_JSON.stat().st_size
    print(f"[완료] {OUT_JSON}  곡선 {len(curves)}/{len(rows)} | {sz/1024:.0f}KB | "
          f"{time.time()-t_all:.0f}s")
    if miss:
        print(f"[주의] 곡선 없음 {len(miss)}: {miss}")


if __name__ == "__main__":
    main()
