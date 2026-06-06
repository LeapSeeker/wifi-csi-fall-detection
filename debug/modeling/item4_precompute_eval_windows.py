"""항목4 평가용 sliding z-SDP precompute (read-only, RPCA 1회 캐싱).

val+test 세션(fall+non-fall)을 full resampled 좌표에서 stride-50 sliding + forward(stride300)
윈도우로 만들어 동결 파이프라인(rpca→acf→sdp→z, = diag_event_sweep._window_z)으로 z-SDP를
1회 계산해 디스크에 저장한다. 15개 모델은 이 캐시로 forward 만 하면 됨(RPCA 중복 제거).

세션 universe: item4 fixed cache 의 split_assignment 에서 split∈{val,test} 인 전 세션(fall+non-fall).
산출: item4/eval_windows.pkl  {filename: {"n_frames", "sweep50":[(s,e,sdp32)...], "forward":[(kind,s,e,sdp32)...]}}
제약(read-only): 원본 CSV·동결파일 무수정. 신규 캐시만.
"""
from __future__ import annotations
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from debug.modeling.diag_event_sweep import _process_session  # noqa: E402  (동결 sliding+z-SDP)

ITEM4 = ROOT / "debug/modeling/diag_out/onset_detector/item4"
CLEANED = ROOT / "data/cleaned"
OUT = ITEM4 / "eval_windows.pkl"
STRIDE = 50


def main():
    fixed = np.load(ITEM4 / "item4_cache_fixed.npz", allow_pickle=True)
    fn = np.array([str(x) for x in fixed["filename"]])
    sp = np.array([str(x) for x in fixed["split_assignment"]])
    # val+test 세션 universe (fall+non-fall)
    sessions = sorted({fn[i] for i in range(len(fn)) if sp[i] in ("val", "test")})
    print(f"[universe] val+test 세션 {len(sessions)} (fixed cache split_assignment 기준)")

    # CSV 경로 매핑
    paths = {}
    missing = []
    for s in sessions:
        p = next(CLEANED.rglob(s), None)
        if p is None:
            missing.append(s)
        else:
            paths[s] = str(p)
    if missing:
        print(f"[경고] CSV 미발견 {len(missing)}: {missing[:5]}")

    store = {}
    t0 = time.time()
    items = [(s, paths[s]) for s in sessions if s in paths]
    n_workers = min(16, max(1, mp.cpu_count() - 2))
    print(f"[병렬] {n_workers} workers × {len(items)} 세션", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_process_session, (p, [STRIDE])): s for (s, p) in items}
        for fut in as_completed(futs):
            s = futs[fut]
            done += 1
            res = fut.result()
            if res.get("error"):
                print(f"  [skip] {s}: {res['error']}", flush=True)
                continue
            sweep = [(int(st), int(en), sdp.astype(np.float32)) for (st, en, sdp) in res["sweep"].get(STRIDE, [])]
            forward = [(kd, int(st), int(en), sdp.astype(np.float32)) for (kd, st, en, sdp) in res["pair"]]
            store[s] = {"n_frames": int(res["n_frames"]), "sweep50": sweep, "forward": forward}
            if done % 40 == 0 or done == len(items):
                print(f"  [{done}/{len(items)}] cached={len(store)} elapsed={time.time()-t0:.0f}s", flush=True)

    OUT.write_bytes(pickle.dumps(store, protocol=4))
    nwin = sum(len(v["sweep50"]) for v in store.values())
    print(f"[완료] {OUT} | 세션 {len(store)} | sweep50 윈도우 {nwin} | {OUT.stat().st_size/1024/1024:.1f}MB | {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
