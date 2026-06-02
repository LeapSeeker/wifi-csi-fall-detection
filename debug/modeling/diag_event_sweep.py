"""event-level operating-point sweep + forward/tail 분리 진단 (재학습 없음).

목적
----
within-subject 모델을 그대로 두고 시스템 후처리(per-window threshold · N연속확정 ·
margin)만으로 event/session 단위 fall recall을 올리고 FAR을 지키는 운영점을 찾는다.
recall 1순위 → N 고정 않고 N∈{1,2} sweep (N=2는 단일-window spike형 낙상 FN 위험 → N=1과 비교).

두 경로를 절대 섞지 않는다
-------------------------
- event sweep(운영점 선택): 실시간 sliding 추론 모사 → tail_window=False (순수 sliding).
  세션 끝 amplitude[-300:] tail을 추가하지 않는다.
- forward/tail 분리 진단(step11): 학습 cache 모사 → tail_window=True (2-window).
  event-level 운영점 metric에는 절대 포함하지 않는다.

기준
----
- 6/4 demo primary = 6-class(pretrained6).
- source_dir = finetune7 summary.json의 source_dir(data/cleaned). cache는 split 재현용일 뿐
  실제 추론 입력이 아니다. held-out CSV를 eval 전용 stride로 재윈도잉해서 추론한다.
- checkpoint: within-subject 우선. GPU full-run 없으면 checkpoints_compare6_cpu/best_operating.pt
  (CPU 30epoch fallback) — 최종 성능 수치 아님, 후처리 sweep 검증용.

재학습/모델/학습 cache 수정 금지. held-out 세션만 사용. 300-frame 불변.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model.preprocessing.acf import N_LAGS
from model.preprocessing.loader import load_safesignal_csv, parse_safesignal_filename
from model.preprocessing.resample import resample_to_100hz
from model.preprocessing.rpca import DEFAULT_MAX_ITER, rpca_sparse
from model.preprocessing.sdp import SUB_STRIDE, SUB_W, W_T, stacked_doppler_profile
from model.preprocessing.window import WINDOW_SIZE, sliding_windows

# ── 경로/상수 (cache builder · summary 기준) ──
SUMMARY_JSON = PROJECT_ROOT / "model" / "finetune" / "cache" / "safesignal_e1234_finetune7.summary.json"
PRETRAINED6_CACHE = PROJECT_ROOT / "model" / "finetune" / "cache" / "safesignal_e1234_pretrained6.npz"
OUT_DIR = PROJECT_ROOT / "debug" / "modeling" / "diag_out"
CKPT_CANDIDATES = [
    # GPU full-run 6-class best_operating.pt 가 추가되면 여기 맨 앞에 둘 것.
    PROJECT_ROOT / "model" / "finetune" / "checkpoints_compare6_cpu" / "best_operating.pt",
]

RX = "both"
TARGET_HZ = 100.0
MAX_GAP_MS = 100.0
RPCA_MAX_ITER = DEFAULT_MAX_ITER  # 200
RPCA_TOL = None
ZSCORE_EPS = 1e-6
FALL_IDX = 0

# split 재현 파라미터 (checkpoint args 와 일치)
SPLIT_SEED = 42
SPLIT_VAL_RATIO = 0.2
SPLIT_TEST_RATIO = 0.2

# sweep grid
SWEEP_STRIDES = [50, 100]
SWEEP_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
SWEEP_N = [1, 2]
SWEEP_MARGINS = [("off", 0.0), ("on_m0.1", 0.1), ("on_m0.2", 0.2)]
FAR_CAP = 0.15

# forward/tail diag
PAIR_STRIDE = None       # → 300
DIAG_THRESHOLDS = [0.20, 0.30]

FALL_ACTIVITIES = {"FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B", "FALL_WALK_F", "FALL_WALK_B"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ── z-score ──
def _zscore(sdp: np.ndarray) -> np.ndarray:
    return ((sdp - sdp.mean()) / (sdp.std() + ZSCORE_EPS)).astype(np.float32)


def _window_z(window: np.ndarray) -> np.ndarray:
    """(300,n_sc) → z-scored SDP (28,20). pipeline.window_to_model_input 동일 경로."""
    sparse = rpca_sparse(window, max_iter=RPCA_MAX_ITER, tol=RPCA_TOL)
    sdp = stacked_doppler_profile(sparse, sub_w=SUB_W, stride=SUB_STRIDE, n_lags=N_LAGS)
    return _zscore(sdp)


# ── 멀티프로세싱 워커: 한 세션 → 모든 stride sliding 윈도우 + pair(forward/tail) ──
def _process_session(arg: tuple[str, list[int]]) -> dict:
    path_str, strides = arg
    path = Path(path_str)
    try:
        raw = load_safesignal_csv(path, rx=RX)
        res = resample_to_100hz(raw.amplitude, raw.timestamps_us, target_hz=TARGET_HZ, max_gap_ms=MAX_GAP_MS)
        amp = res.amplitude
        n = int(amp.shape[0])
        out: dict = {
            "filename": path.name, "n_frames": n,
            "orig_rate": float(res.original_rate_hz), "error": None,
            "sweep": {}, "pair": [],
        }
        if n < WINDOW_SIZE:
            out["error"] = f"n<300 (n={n})"
            return out

        # event sweep: tail_window=False sliding
        for s in strides:
            wins = sliding_windows(amp, window_size=WINDOW_SIZE, stride=s,
                                   drop_last=True, tail_window=False, pad_short=False)
            n_w = wins.shape[0]
            starts = [k * s for k in range(n_w)]
            items = []
            for st, w in zip(starts, wins):
                items.append((st, st + WINDOW_SIZE, _window_z(w)))
            out["sweep"][s] = items

        # pair: tail_window=True, stride=300 (학습 cache 정의 재현)
        wins = sliding_windows(amp, window_size=WINDOW_SIZE, stride=PAIR_STRIDE,
                               drop_last=True, tail_window=True, pad_short=False)
        stride300 = WINDOW_SIZE
        n_fwd = 1 + (n - WINDOW_SIZE) // stride300
        fwd_starts = [k * stride300 for k in range(n_fwd)]
        starts = list(fwd_starts)
        kinds = ["forward"] * n_fwd
        last_end = fwd_starts[-1] + WINDOW_SIZE
        if last_end < n:
            starts.append(n - WINDOW_SIZE)
            kinds.append("tail")
        for kd, st, w in zip(kinds, starts, wins):
            out["pair"].append((kd, st, st + WINDOW_SIZE, _window_z(w)))
        return out
    except Exception as exc:  # noqa
        return {"filename": path.name, "n_frames": 0, "orig_rate": float("nan"),
                "error": repr(exc), "sweep": {}, "pair": []}


# ── held-out 세션 재현 ──
def reproduce_heldout() -> dict:
    """train.py 의 SafeSignalDataset + split_safesignal_within_subject 를 그대로 재실행해
    held-out(test) 세션 파일명 집합을 얻는다. (재구현 금지 — 동일 함수 호출)"""
    from model.finetune.train import SafeSignalDataset, split_safesignal_within_subject

    ds = SafeSignalDataset.from_npz(PRETRAINED6_CACHE)
    tr, va, te = split_safesignal_within_subject(
        ds, val_ratio=SPLIT_VAL_RATIO, test_ratio=SPLIT_TEST_RATIO, seed=SPLIT_SEED
    )

    def sess(subset) -> set[str]:
        return {ds.filenames[i] for i in subset.indices}

    test_sessions = sess(te)
    train_sessions = sess(tr)
    val_sessions = sess(va)
    return {
        "classes": None,  # set by caller from cache
        "n_total_sessions": len(set(ds.filenames)),
        "test": sorted(test_sessions),
        "train": sorted(train_sessions),
        "val": sorted(val_sessions),
        "dataset": ds,
    }


# ── 세션 메타 ──
def _meta(filename: str):
    return parse_safesignal_filename(filename)


def _is_fall(filename: str) -> bool:
    return _meta(filename).activity in FALL_ACTIVITIES


# ── N연속 확정 ──
def _fires(positive_seq: list[bool], n: int) -> bool:
    if n <= 1:
        return any(positive_seq)
    run = 0
    for p in positive_seq:
        run = run + 1 if p else 0
        if run >= n:
            return True
    return False


@dataclass
class ConfigResult:
    threshold_min: float
    N: int
    margin_mode: str
    margin_value: float
    stride: int
    event_recall: float
    event_FAR: float
    event_F1: float
    TP: int
    FP: int
    FN: int
    TN: int
    confirmation_extra_latency_s: float
    window_end_latency_s: float


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="held-out 세션 수 제한 (0=전체, dry-run용)")
    ap.add_argument("--workers", type=int, default=0, help="0=cpu_count-1")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 80)
    log("event-level operating-point sweep + forward/tail 분리 진단 (재학습 없음)")
    log("=" * 80)

    # ── 기준 경로/메타 ──
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    source_dir = Path(summary["source_dir"])
    log(f"source_dir (finetune7 summary): {source_dir}")
    log(f"pretrained6 cache (split 재현용): {PRETRAINED6_CACHE.name}")

    cache = np.load(PRETRAINED6_CACHE, allow_pickle=True)
    cache_classes = [str(c) for c in cache["classes"].tolist()] if "classes" in cache.files else None
    cache_policy = str(cache["class_policy"]) if "class_policy" in cache.files else None
    log(f"cache classes={cache_classes} class_policy={cache_policy}")

    # ── checkpoint ──
    import torch
    from model.pretrained.model import CNNGRUAttention

    ckpt_path = None
    for c in CKPT_CANDIDATES:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        log("FAIL: checkpoint 미발견. 탐색 경로:")
        for c in CKPT_CANDIDATES:
            log(f"  {c}")
        return 1
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_classes = ckpt.get("classes")
    ck_policy = ckpt.get("class_policy")
    log(f"\ncheckpoint: {ckpt_path}")
    log(f"  classes={ck_classes} class_policy={ck_policy} threshold(saved)={ckpt.get('threshold')}")
    is_gpu_fullrun = "checkpoints_compare6_cpu" not in str(ckpt_path)
    log(f"  해석: {'GPU full-run' if is_gpu_fullrun else 'CPU 30epoch fallback → 후처리 sweep 검증용, 최종 성능 수치 아님'}")

    # class metadata 검증 (완화)
    warn_meta = []
    if cache_classes and ck_classes and cache_classes != ck_classes:
        log(f"FAIL: cache classes {cache_classes} != checkpoint classes {ck_classes}")
        return 1
    if cache_policy and ck_policy and cache_policy != ck_policy:
        log(f"FAIL: cache policy {cache_policy} != checkpoint policy {ck_policy}")
        return 1
    n_out = ckpt["model"]["classifier.1.weight"].shape[0]
    if n_out != 6:
        log(f"FAIL: 모델 output dim={n_out} != 6 (6-class pretrained6 아님)")
        return 1
    eff_classes = ck_classes or cache_classes
    if eff_classes is None:
        warn_meta.append("classes metadata 부재 → output dim==6, fall index==0 만 확인하고 진행")
    elif eff_classes[FALL_IDX] != "fall":
        log(f"FAIL: classes[0]={eff_classes[FALL_IDX]} != 'fall'")
        return 1
    log(f"  output dim={n_out}, fall index={FALL_IDX} ('fall') 확인")

    model = CNNGRUAttention(n_classes=6)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    log(f"  device={device}")

    # ── held-out 재현 ──
    log("\n" + "=" * 80)
    log("held-out(test) 세션 재현 — manifest 없음 → split_safesignal_within_subject 재실행")
    log("=" * 80)
    log("탐색한 manifest 위치(부재 확인): checkpoint metadata(args만 존재), "
        "checkpoints_compare6_cpu/{history.json, within_subject_test_report.json}, runs/·wandb 없음")
    rep = reproduce_heldout()
    ds = rep["dataset"]
    log(f"split: within_subject seed={SPLIT_SEED} val_ratio={SPLIT_VAL_RATIO} test_ratio={SPLIT_TEST_RATIO}")
    log(f"세션 수: total={rep['n_total_sessions']} train={len(rep['train'])} "
        f"val={len(rep['val'])} test(held-out)={len(rep['test'])}")
    log("경고: 0b25a49 within-subject split과 동일 가정에 의존(동일 seed/ratio/cache) → 운영점 잠정.")

    test_sessions = rep["test"]
    if args.limit > 0:
        test_sessions = test_sessions[: args.limit]
        log(f"[dry-run] held-out 세션 {len(test_sessions)}개로 제한")

    # held-out 구성
    fall_sess = [f for f in test_sessions if _is_fall(f)]
    nonfall_sess = [f for f in test_sessions if not _is_fall(f)]
    by_act: dict[str, int] = {}
    for f in test_sessions:
        a = _meta(f).activity
        by_act[a] = by_act.get(a, 0) + 1
    log(f"held-out fall 세션={len(fall_sess)} non-fall 세션={len(nonfall_sess)} "
        f"(RUN 제외 pretrained6 데모 범위)")
    log(f"held-out activity 분포: {dict(sorted(by_act.items()))}")
    if len(nonfall_sess) < 60:
        log(f"주의: 비낙상 held-out 세션 수가 적음({len(nonfall_sess)}) → event-FAR 분산 큼.")

    # held-out CSV 경로 확인
    paths = []
    missing = []
    for f in test_sessions:
        p = source_dir / f
        if p.exists():
            paths.append(p)
        else:
            missing.append(f)
    if missing:
        log(f"주의: source_dir에서 누락된 held-out CSV {len(missing)}개 (예: {missing[:3]})")
    log(f"처리할 held-out CSV: {len(paths)}")

    # ── 전처리 (멀티프로세싱 RPCA) ──
    log("\n" + "=" * 80)
    log("held-out 세션 재윈도잉 + RPCA→ACF→SDP→z-score (event sweep=sliding, pair=2-window)")
    log("=" * 80)
    n_workers = args.workers if args.workers > 0 else max(1, __import__("multiprocessing").cpu_count() - 1)
    log(f"workers={n_workers} strides(sweep)={SWEEP_STRIDES} pair_stride=300(tail_window=True)")
    sessions: dict[str, dict] = {}
    work = [(str(p), SWEEP_STRIDES) for p in paths]
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_process_session, w) for w in work]
        for fut in as_completed(futs):
            r = fut.result()
            sessions[r["filename"]] = r
            done += 1
            if done % 25 == 0:
                log(f"  [progress] {done}/{len(work)} 세션 처리")
    skipped = {f: r["error"] for f, r in sessions.items() if r["error"]}
    if skipped:
        log(f"  skip {len(skipped)} 세션: {list(skipped.items())[:5]}")

    # ── 모델 추론 (모든 윈도우 배치) ──
    log("\n모델 추론 (배치 softmax → fall_prob, second_prob=max non-fall)")
    flat_z: list[np.ndarray] = []
    refs: list[tuple] = []  # (filename, 'sweep', stride, widx) | (filename, 'pair', None, widx)
    for f, r in sessions.items():
        if r["error"]:
            continue
        for s, items in r["sweep"].items():
            for wi, (st, en, z) in enumerate(items):
                flat_z.append(z)
                refs.append((f, "sweep", s, wi))
        for wi, (kd, st, en, z) in enumerate(r["pair"]):
            flat_z.append(z)
            refs.append((f, "pair", None, wi))

    probs = np.zeros((len(flat_z), 2), dtype=np.float32)  # [fall_prob, second_prob]
    if flat_z:
        X = np.stack(flat_z, axis=0)[:, None, :, :]  # (M,1,28,20)
        bs = 512
        with torch.no_grad():
            for i in range(0, len(X), bs):
                t = torch.from_numpy(X[i : i + bs]).to(device)
                logits = model(t)
                sm = torch.softmax(logits, dim=1).cpu().numpy()
                fall_p = sm[:, FALL_IDX]
                non = np.delete(sm, FALL_IDX, axis=1)
                second_p = non.max(axis=1)
                probs[i : i + len(sm), 0] = fall_p
                probs[i : i + len(sm), 1] = second_p
    log(f"  추론 윈도우 총 {len(flat_z)}개")

    # 확률을 세션 구조에 부착
    # sweep_probs[f][stride] = list of dict(start,end,fall,second)
    sweep_probs: dict[str, dict[int, list[dict]]] = {}
    pair_probs: dict[str, list[dict]] = {}
    for (f, kind, s, wi), (fall_p, second_p) in zip(refs, probs):
        if kind == "sweep":
            items = sessions[f]["sweep"][s]
            st, en, _ = items[wi]
            sweep_probs.setdefault(f, {}).setdefault(s, []).append(
                {"start": st, "end": en, "fall": float(fall_p), "second": float(second_p)})
        else:
            kd, st, en, _ = sessions[f]["pair"][wi]
            pair_probs.setdefault(f, []).append(
                {"kind": kd, "start": st, "end": en, "fall": float(fall_p), "second": float(second_p)})
    for f in sweep_probs:
        for s in sweep_probs[f]:
            sweep_probs[f][s].sort(key=lambda d: d["start"])

    valid_sessions = [f for f in test_sessions if f in sessions and not sessions[f]["error"]]
    valid_fall = [f for f in valid_sessions if _is_fall(f)]
    valid_nonfall = [f for f in valid_sessions if not _is_fall(f)]

    # ── sanity check: held-out window-level recall (pair 윈도우, threshold 0.1) ──
    log("\n" + "=" * 80)
    log("sanity check — held-out fall window-level recall @ threshold 0.1 (학습 cache 동일 2-window)")
    log("=" * 80)
    fall_win = []
    for f in valid_fall:
        for d in pair_probs.get(f, []):
            fall_win.append(d["fall"])
    if fall_win:
        win_recall = float(np.mean([p >= 0.1 for p in fall_win]))
        log(f"  fall window 수={len(fall_win)}  window recall(@0.1)={win_recall:.3f}")
        log(f"  참고: within_subject_test_report.json fall_recall=0.736 (tp=106/144), task hint R≈0.674")
        if abs(win_recall - 0.736) > 0.08 and abs(win_recall - 0.674) > 0.08:
            log(f"  경고: window recall {win_recall:.3f} 가 report(0.736)/hint(0.674) 모두와 >0.08 차이 "
                f"→ split/checkpoint 불일치 가능성")
        else:
            log(f"  OK: report/hint 근처 → split·checkpoint 재현 일관")
    else:
        log("  경고: fall window 없음 — held-out fall 세션 확인 필요")

    # ── event sweep ──
    log("\n" + "=" * 80)
    log("event-level sweep (tail_window=False sliding; tail 윈도우 미포함)")
    log("=" * 80)
    log(f"sweep grid: threshold_min={SWEEP_THRESHOLDS} N={SWEEP_N} "
        f"margin={[m[0] for m in SWEEP_MARGINS]} stride={SWEEP_STRIDES}")
    log("주의: event_FAR 는 train.py window-level FAR 와 다른 지표(세션 단위 1회+ 발화).")

    results: list[ConfigResult] = []
    for stride in SWEEP_STRIDES:
        for t in SWEEP_THRESHOLDS:
            for (mmode, mval) in SWEEP_MARGINS:
                for N in SWEEP_N:
                    def session_fires(f: str) -> bool:
                        seq = sweep_probs.get(f, {}).get(stride, [])
                        pos = []
                        for d in seq:
                            ok = d["fall"] >= t
                            if mmode != "off":
                                ok = ok and (d["fall"] - d["second"]) >= mval
                            pos.append(ok)
                        return _fires(pos, N)

                    tp = sum(session_fires(f) for f in valid_fall)
                    fp = sum(session_fires(f) for f in valid_nonfall)
                    fn = len(valid_fall) - tp
                    tn = len(valid_nonfall) - fp
                    recall = tp / len(valid_fall) if valid_fall else float("nan")
                    far = fp / len(valid_nonfall) if valid_nonfall else float("nan")
                    prec = tp / (tp + fp) if (tp + fp) else 0.0
                    f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) else 0.0
                    results.append(ConfigResult(
                        threshold_min=t, N=N, margin_mode=mmode, margin_value=mval, stride=stride,
                        event_recall=recall, event_FAR=far, event_F1=f1,
                        TP=tp, FP=fp, FN=fn, TN=tn,
                        confirmation_extra_latency_s=(N - 1) * stride / 100.0,
                        window_end_latency_s=(WINDOW_SIZE + (N - 1) * stride) / 100.0,
                    ))

    # 결과표 CSV
    sweep_csv = OUT_DIR / "event_sweep_results.csv"
    cols = ["threshold_min", "N", "margin_mode", "margin_value", "stride", "event_recall",
            "event_FAR", "event_F1", "TP", "FP", "FN", "TN",
            "confirmation_extra_latency_s", "window_end_latency_s"]
    lines = [",".join(cols)]
    for r in results:
        d = asdict(r)
        lines.append(",".join(str(round(d[c], 4) if isinstance(d[c], float) else d[c]) for c in cols))
    sweep_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── 운영점 선택 ──
    log("\n" + "-" * 80)
    log("운영점 선택: FAR<=0.15 하 event_recall 최대 (tie→FAR↓, tie→N=1·stride=50)")
    log("-" * 80)
    feasible = [r for r in results if not np.isnan(r.event_FAR) and r.event_FAR <= FAR_CAP]
    preferred = None
    if feasible:
        preferred = sorted(feasible, key=lambda r: (
            -r.event_recall, r.event_FAR, r.N, r.stride, -r.event_F1))[0]
        log(f"  후보(FAR<=0.15) {len(feasible)}개 중 1순위:")
    else:
        log("  FAR<=0.15 만족 config 없음 → 차선: FAR 최소 중 recall 최대")
        preferred = sorted(results, key=lambda r: (r.event_FAR, -r.event_recall))[0]

    def fmt(r: ConfigResult) -> str:
        return (f"t={r.threshold_min} N={r.N} margin={r.margin_mode} stride={r.stride} | "
                f"recall={r.event_recall:.3f} FAR={r.event_FAR:.3f} F1={r.event_F1:.3f} | "
                f"TP={r.TP} FP={r.FP} FN={r.FN} TN={r.TN} | "
                f"+lat={r.confirmation_extra_latency_s:.2f}s end_lat={r.window_end_latency_s:.2f}s")
    log(f"  ★ {fmt(preferred)}")

    # N=1 vs N=2 비교 (동일 t/margin/stride)
    log("\n  [N=1 vs N=2 비교 — spike형 낙상 FN 위험 점검] (margin off)")
    log("   stride  t     N=1 recall/FAR        N=2 recall/FAR        Δrecall")
    for stride in SWEEP_STRIDES:
        for t in SWEEP_THRESHOLDS:
            r1 = next(r for r in results if r.stride == stride and r.threshold_min == t and r.N == 1 and r.margin_mode == "off")
            r2 = next(r for r in results if r.stride == stride and r.threshold_min == t and r.N == 2 and r.margin_mode == "off")
            log(f"   {stride:4d}  {t:.2f}  {r1.event_recall:.3f}/{r1.event_FAR:.3f}        "
                f"{r2.event_recall:.3f}/{r2.event_FAR:.3f}        {r2.event_recall - r1.event_recall:+.3f}")

    # top configs by recall (FAR<=0.15)
    log("\n  [FAR<=0.15 config 상위 8 by recall]")
    for r in sorted(feasible, key=lambda r: (-r.event_recall, r.event_FAR))[:8]:
        log(f"    {fmt(r)}")

    # ── tradeoff plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for stride, mk in zip(SWEEP_STRIDES, ["o", "s"]):
        for N, cl in zip(SWEEP_N, ["tab:blue", "tab:red"]):
            rs = [r for r in results if r.stride == stride and r.N == N]
            ax.scatter([r.event_FAR for r in rs], [r.event_recall for r in rs],
                       marker=mk, c=cl, alpha=0.6, label=f"stride={stride} N={N}")
    ax.axvline(FAR_CAP, color="k", ls="--", lw=1, label="FAR=0.15 cap")
    if preferred:
        ax.scatter([preferred.event_FAR], [preferred.event_recall], marker="*", s=300,
                   c="gold", edgecolors="k", zorder=5, label="preferred")
    ax.set_xlabel("event_FAR"); ax.set_ylabel("event_recall")
    ax.set_title("event-level recall vs FAR sweep (held-out, pretrained6)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "event_recall_far_tradeoff.png", dpi=120); plt.close(fig)

    # ── preferred 세션별 확률 시퀀스 CSV ──
    seq_csv = OUT_DIR / "preferred_session_prob_sequences.csv"
    seq_cols = ["session", "activity", "subject", "env", "trial", "stride", "window_start_frame",
                "window_end_frame", "window_kind", "fall_prob", "second_prob", "margin", "positive"]
    slines = [",".join(seq_cols)]
    if preferred:
        ps, pt, pm, pmv, pN = preferred.stride, preferred.threshold_min, preferred.margin_mode, preferred.margin_value, preferred.N
        for f in valid_sessions:
            m = _meta(f)
            for d in sweep_probs.get(f, {}).get(ps, []):
                marg = d["fall"] - d["second"]
                pos = d["fall"] >= pt and (pm == "off" or marg >= pmv)
                slines.append(",".join(str(x) for x in [
                    f, m.activity, m.subject, m.environment, m.trial, ps,
                    d["start"], d["end"], "sliding",
                    round(d["fall"], 4), round(d["second"], 4), round(marg, 4), int(pos)]))
    seq_csv.write_text("\n".join(slines) + "\n", encoding="utf-8")

    # ── step11: forward/tail 분리 진단 (event sweep과 완전 별개, tail_window=True) ──
    log("\n" + "=" * 80)
    log("[step11] forward/tail 분리 진단 — 학습 cache 2-window(tail_window=True, stride=300)")
    log("=" * 80)
    log("주의: 이 결과는 step3-a(tail down-weight) 판단용. event-level 운영점 metric에 미포함.")
    diag_rows = []
    for t in DIAG_THRESHOLDS:
        # window-kind별 recall/FN (fall 세션)
        fwd_pos = fwd_tot = tail_pos = tail_tot = 0
        fwd_only = tail_only = 0  # fall 세션 기준
        for f in valid_fall:
            ds_ = pair_probs.get(f, [])
            fwd = [d for d in ds_ if d["kind"] == "forward"]
            tl = [d for d in ds_ if d["kind"] == "tail"]
            fwd_p = any(d["fall"] >= t for d in fwd)
            tail_p = any(d["fall"] >= t for d in tl)
            for d in fwd:
                fwd_tot += 1; fwd_pos += int(d["fall"] >= t)
            for d in tl:
                tail_tot += 1; tail_pos += int(d["fall"] >= t)
            if fwd_p and not tail_p:
                fwd_only += 1
            if tail_p and not fwd_p:
                tail_only += 1
        # 비낙상 FP by kind
        fwd_fp = tail_fp = fwd_fp_tot = tail_fp_tot = 0
        for f in valid_nonfall:
            ds_ = pair_probs.get(f, [])
            for d in ds_:
                if d["kind"] == "forward":
                    fwd_fp_tot += 1; fwd_fp += int(d["fall"] >= t)
                else:
                    tail_fp_tot += 1; tail_fp += int(d["fall"] >= t)
        fwd_recall = fwd_pos / fwd_tot if fwd_tot else float("nan")
        tail_recall = tail_pos / tail_tot if tail_tot else float("nan")
        log(f"\n threshold={t}")
        log(f"   forward window recall={fwd_recall:.3f} ({fwd_pos}/{fwd_tot})  FN={fwd_tot - fwd_pos}  "
            f"비낙상 FP={fwd_fp}/{fwd_fp_tot}")
        log(f"   tail    window recall={tail_recall:.3f} ({tail_pos}/{tail_tot})  FN={tail_tot - tail_pos}  "
            f"비낙상 FP={tail_fp}/{tail_fp_tot}")
        log(f"   forward-positive & tail-negative fall 세션={fwd_only}")
        log(f"   tail-positive & forward-negative fall 세션(tail 단독 구제)={tail_only}")
        # CSV rows (window_kind별)
        diag_rows.append([t, "forward", round(fwd_recall, 4), fwd_tot - fwd_pos, fwd_fp, tail_only, fwd_only])
        diag_rows.append([t, "tail", round(tail_recall, 4), tail_tot - tail_pos, tail_fp, tail_only, fwd_only])
        # 해석
        if not np.isnan(tail_recall):
            if tail_recall < fwd_recall and tail_only <= max(1, int(0.02 * len(valid_fall))):
                log(f"   해석: tail recall<forward 이고 tail 단독 구제={tail_only}(미미) → step3-a(tail down-weight) 안전.")
            else:
                log(f"   해석: tail 단독 구제={tail_only} 무시 못함 → tail down-weight 신중.")

    diag_csv = OUT_DIR / "forward_tail_split_diag.csv"
    dcols = ["threshold", "window_kind", "window_recall", "FN", "FP_nonfall",
             "tail_only_rescue_sessions", "forward_only_sessions"]
    dlines = [",".join(dcols)] + [",".join(str(x) for x in row) for row in diag_rows]
    diag_csv.write_text("\n".join(dlines) + "\n", encoding="utf-8")

    # ── 산출물 ──
    log("\n" + "=" * 80)
    log("[산출물]  (debug/modeling/diag_out/, 비추적)")
    log("=" * 80)
    log(f"  sweep 결과표      : {sweep_csv}")
    log(f"  forward/tail 진단 : {diag_csv}")
    log(f"  tradeoff plot     : {OUT_DIR / 'event_recall_far_tradeoff.png'}")
    log(f"  preferred 시퀀스  : {seq_csv}")
    log(f"  checkpoint_used   : {ckpt_path}")
    if warn_meta:
        for w in warn_meta:
            log(f"  [경고] {w}")
    log("\n결론(잠정): held-out event 수가 적어 1위 config는 노이즈 가능 → '잠정 운영점'. "
        "추가 held-out/데모 리허설로 확정 필요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
