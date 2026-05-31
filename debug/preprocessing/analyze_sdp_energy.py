"""SafeSignal CSV의 z-score 적용 *전* SDP energy를 정식 전처리 파이프라인으로 산출.

목적
----
낙상 오탐 억제(no_motion / low-energy static gate) 설계를 위해, 모델 입력 직전
구조에 가장 가까운 SDP energy 분포를 활동별로 비교한다. ai-workspace 탐색용
분석(RPCA max_iter=30, raw timestamp 300 frame 균등 샘플링)과 달리, 이 스크립트는
정식 SafeSignal 경로를 그대로 사용한다:

    load_safesignal_csv → resample_to_100hz → sliding_windows
      → rpca_sparse → stacked_doppler_profile → (z-score 적용 직전) energy 계산

주의: 정식 추론/학습은 window_to_model_input() 안에서 SDP 직후 global z-score를
적용한다(pipeline.py). 이 스크립트는 그 z-score 라인 직전까지만 재현하여 energy를
보존한다. 절대 임계값을 즉시 hard gate로 코드에 박지 말 것 — fall 세션 energy p5와
비교 후 확정한다(STATE D-020 / FALSE_POSITIVE_IMPROVEMENT_NOTES 7.5 참조).

산출 지표 (윈도우 단위, 모두 z-score 전)
----------------------------------------
  sdp_mean_abs   : mean(|SDP|)              — 1차 gate 후보
  sdp_fro        : ||SDP||_F                — 전체 energy
  sdp_std        : SDP.std()
  sdp_max_abs    : max(|SDP|)
  sparse_ratio   : ||S||_F / ||window||_F   — RPCA sparse energy 비율
  raw_std        : window.std()             — 리샘플된 raw 진폭 분산
  raw_delta_mean : mean(|diff(window, axis=0)|)  — frame-to-frame 평균 변화량

파일 단위 metadata (가능하면 포함, report-only)
  pair_dt_p50/p95/p99/max_us  : CSV pair_dt_us 컬럼(110컬럼 CSV만; 없으면 None)
  original_rate_hz            : resample 입력 추정 rate
  max_gap_ms / gap_count      : resample 입력 timestamp 최대 gap / 초과 개수
주의: pair_dt/ts_gap은 절대 hard reject 기준으로 쓰지 않는다(STATE D-025, 노트 §8).

출력
----
  - 활동별 p50/p95/p99/min/max
  - no_motion p95/p99 기준 각 활동의 초과 비율
  - --out 지정 시 윈도우 단위 결과를 CSV 또는 JSON으로 저장 (확장자로 판별)

실행 (PROJECT_ROOT = wifi-csi-fall-detection, scipy/pandas/numpy 환경 필요)
-------------------------------------------------------------------------
    # 전체 data/raw 분석 (느림 — RPCA 200 iter × 모든 윈도우)
    python debug/preprocessing/analyze_sdp_energy.py

    # 특정 활동만 (여러 개 지정 가능)
    python debug/preprocessing/analyze_sdp_energy.py --activity NO_MOTION WALK SIT_STD

    # 빠른 점검: 파일당 윈도우 수 제한 + 단일 프로세스
    python debug/preprocessing/analyze_sdp_energy.py --activity NO_MOTION \
        --limit-windows-per-file 8 --workers 1

    # 긴 단일 프로세스 실행에서 윈도우 진행률 표시
    python debug/preprocessing/analyze_sdp_energy.py --activity NO_MOTION \
        --workers 1 --progress-every 10

    # 결과 저장
    python debug/preprocessing/analyze_sdp_energy.py --out sdp_energy.csv
    python debug/preprocessing/analyze_sdp_energy.py --out sdp_energy.json

    # 그룹 기준: activity(기본, FALL_SIT_F 등 raw 코드) 또는 class(fall 등 통합)
    python debug/preprocessing/analyze_sdp_energy.py --group-mode class

Windows/macOS 주의: --workers>1 이면 ProcessPoolExecutor(spawn)를 사용하므로
반드시 ``if __name__ == "__main__":`` 진입점(본 스크립트는 이미 보장)으로 실행할 것.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from model.preprocessing.loader import load_safesignal_csv, parse_safesignal_filename
from model.preprocessing.resample import resample_to_100hz
from model.preprocessing.rpca import DEFAULT_MAX_ITER, rpca_sparse
from model.preprocessing.sdp import stacked_doppler_profile
from model.preprocessing.window import WINDOW_SIZE, sliding_windows

# 3초 윈도우 / 1초 stride (100Hz 기준)
DEFAULT_STRIDE = 100

# 윈도우 단위 SDP energy 지표 (z-score 전) — 출력/집계 순서 고정
SDP_METRICS = ["sdp_mean_abs", "sdp_fro", "sdp_std", "sdp_max_abs"]
RAW_METRICS = ["sparse_ratio", "raw_std", "raw_delta_mean"]
WINDOW_METRICS = SDP_METRICS + RAW_METRICS

# no_motion 초과 비율 보고 대상 (gate 후보 지표)
EXCEED_METRICS = ["sdp_mean_abs", "sdp_std", "sparse_ratio"]

# FALL_SIT_F 등 raw 활동 코드 → 통합 클래스 (D-022/D-023 명명)
_CLASS_MAP = {
    "SIT_STD": "sit_stand",
    "LIE": "lying",
    "WALK": "walking",
    "STAND": "standing",
    "RUN": "running",
    "PICK": "picking",
    "NO_MOTION": "no_motion",
}


def activity_to_class(activity: str) -> str:
    """raw 활동 코드를 통합 클래스명으로. FALL_* → fall."""
    act = activity.upper()
    if act.startswith("FALL"):
        return "fall"
    return _CLASS_MAP.get(act, act.lower())


def _is_no_motion_group(group_key: str) -> bool:
    return group_key.upper() == "NO_MOTION" or group_key == "no_motion"


# ─── 윈도우 단위 지표 ────────────────────────────────────────────────────────

def _window_metrics(window: np.ndarray, rpca_max_iter: int, rpca_tol: float | None) -> dict:
    """단일 윈도우 (300, n_sc) → z-score 전 SDP energy + raw 지표 dict.

    정식 window_to_model_input()의 z-score 라인(pipeline.py) 직전까지 재현:
      rpca_sparse → stacked_doppler_profile → (여기서 z-score 전 energy 계산).
    """
    sparse = rpca_sparse(window, max_iter=rpca_max_iter, tol=rpca_tol)
    sdp = stacked_doppler_profile(sparse)  # (28, 20) — z-score 적용 전

    abs_sdp = np.abs(sdp)
    win_fro = float(np.linalg.norm(window))
    sparse_fro = float(np.linalg.norm(sparse))

    return {
        "sdp_mean_abs": float(abs_sdp.mean()),
        "sdp_fro": float(np.linalg.norm(sdp)),
        "sdp_std": float(sdp.std()),
        "sdp_max_abs": float(abs_sdp.max()),
        "sparse_ratio": float(sparse_fro / win_fro) if win_fro > 0 else 0.0,
        "raw_std": float(window.std()),
        "raw_delta_mean": float(np.abs(np.diff(window, axis=0)).mean())
        if window.shape[0] >= 2 else 0.0,
    }


def _file_pair_dt_meta(path: Path) -> dict:
    """CSV의 pair_dt_us 컬럼(110컬럼 신규 CSV만) → p50/p95/p99/max. legacy면 None."""
    meta = {
        "pair_dt_p50_us": None,
        "pair_dt_p95_us": None,
        "pair_dt_p99_us": None,
        "pair_dt_max_us": None,
    }
    try:
        head = pd.read_csv(path, nrows=0)
        if "pair_dt_us" not in head.columns:
            return meta
        pair_dt = pd.read_csv(path, usecols=["pair_dt_us"])["pair_dt_us"]
        pair_dt = pair_dt.dropna().to_numpy(dtype=float)
        if pair_dt.size:
            meta["pair_dt_p50_us"] = float(np.percentile(pair_dt, 50))
            meta["pair_dt_p95_us"] = float(np.percentile(pair_dt, 95))
            meta["pair_dt_p99_us"] = float(np.percentile(pair_dt, 99))
            meta["pair_dt_max_us"] = float(pair_dt.max())
    except Exception:
        pass
    return meta


# ─── 파일 단위 워커 (ProcessPoolExecutor picklable) ─────────────────────────

def _process_file(args: tuple) -> tuple[str, list[dict] | None, str | None]:
    """단일 CSV → 윈도우 단위 metric record 목록.

    Returns (filename, records | None, error_msg | None).
    records 각 원소는 윈도우 지표 + 파일 metadata(파일 단위 broadcast).
    """
    (
        path_str,
        group_mode,
        window_size,
        stride,
        max_gap_ms,
        limit_windows,
        rpca_max_iter,
        rpca_tol,
        progress_every,
    ) = args
    path = Path(path_str)
    try:
        meta = parse_safesignal_filename(path)
        raw = load_safesignal_csv(path, rx="both")
        res = resample_to_100hz(
            raw.amplitude, raw.timestamps_us, max_gap_ms=max_gap_ms
        )
        windows = sliding_windows(
            res.amplitude,
            window_size=window_size,
            stride=stride,
            drop_last=True,
        )
        if limit_windows is not None and windows.shape[0] > limit_windows:
            windows = windows[:limit_windows]

        group_key = (
            activity_to_class(meta.activity) if group_mode == "class" else meta.activity
        )
        pair_meta = _file_pair_dt_meta(path)
        file_ctx = {
            "file": path.name,
            "activity": meta.activity,
            "group": group_key,
            "environment": meta.environment,
            "subject": meta.subject,
            "trial": meta.trial,
            "original_rate_hz": float(res.original_rate_hz),
            "max_gap_ms": float(res.max_gap_us) / 1000.0,
            "gap_count": int(res.gap_count),
            **pair_meta,
        }

        records: list[dict] = []
        total_windows = int(windows.shape[0])
        if progress_every:
            print(f"    ▶ {path.name}: {total_windows} windows 시작", flush=True)
        for i, w in enumerate(windows):
            m = _window_metrics(w, rpca_max_iter=rpca_max_iter, rpca_tol=rpca_tol)
            rec = {**file_ctx, "window_idx": i, **m}
            records.append(rec)
            if progress_every and ((i + 1) % progress_every == 0 or (i + 1) == total_windows):
                print(
                    f"    {path.name}: window {i + 1}/{total_windows}",
                    flush=True,
                )
        return path.name, records, None
    except Exception as e:  # noqa: BLE001 — 파일 1개 실패가 전체를 막지 않도록
        return path.name, None, repr(e)


# ─── 집계 ────────────────────────────────────────────────────────────────────

def _percentiles(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def aggregate(records: list[dict]) -> dict[str, dict[str, dict]]:
    """group → metric → {n,p50,p95,p99,min,max}."""
    by_group: dict[str, list[dict]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    summary: dict[str, dict[str, dict]] = {}
    for group, recs in sorted(by_group.items()):
        per_metric: dict[str, dict] = {}
        for metric in WINDOW_METRICS:
            vals = np.asarray([r[metric] for r in recs], dtype=float)
            if vals.size:
                per_metric[metric] = _percentiles(vals)
        summary[group] = {"n_windows": len(recs), "metrics": per_metric}
    return summary


def exceed_ratios(records: list[dict], summary: dict) -> dict | None:
    """no_motion p95/p99 기준, 각 group의 초과 윈도우 비율."""
    nm_group = next((g for g in summary if _is_no_motion_group(g)), None)
    if nm_group is None:
        return None

    nm_metrics = summary[nm_group]["metrics"]
    by_group: dict[str, list[dict]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    out: dict[str, dict] = {"reference_group": nm_group, "groups": {}}
    for group, recs in sorted(by_group.items()):
        per_metric: dict[str, dict] = {}
        for metric in EXCEED_METRICS:
            if metric not in nm_metrics:
                continue
            p95 = nm_metrics[metric]["p95"]
            p99 = nm_metrics[metric]["p99"]
            vals = np.asarray([r[metric] for r in recs], dtype=float)
            per_metric[metric] = {
                "p95_thresh": p95,
                "p99_thresh": p99,
                "ratio_gt_p95": float((vals > p95).mean()),
                "ratio_gt_p99": float((vals > p99).mean()),
            }
        out["groups"][group] = per_metric
    return out


# ─── 출력 ────────────────────────────────────────────────────────────────────

def print_report(summary: dict, exceed: dict | None) -> None:
    print("\n=== 활동별 z-score 전 SDP energy 분포 (3s window / 1s stride) ===")
    for group, info in summary.items():
        print(f"\n[{group}]  n_windows={info['n_windows']}")
        for metric in WINDOW_METRICS:
            s = info["metrics"].get(metric)
            if s is None:
                continue
            print(
                f"  {metric:<14} p50={s['p50']:.5f} p95={s['p95']:.5f} "
                f"p99={s['p99']:.5f} min={s['min']:.5f} max={s['max']:.5f}"
            )

    if exceed is None:
        print("\n[no_motion 초과 비율] NO_MOTION 그룹이 없어 생략")
        return

    print(f"\n=== no_motion({exceed['reference_group']}) p95/p99 초과 윈도우 비율 ===")
    for group, per_metric in exceed["groups"].items():
        print(f"\n[{group}]")
        for metric in EXCEED_METRICS:
            d = per_metric.get(metric)
            if d is None:
                continue
            print(
                f"  {metric:<14} >p95({d['p95_thresh']:.5f})={d['ratio_gt_p95']:.3f}  "
                f">p99({d['p99_thresh']:.5f})={d['ratio_gt_p99']:.3f}"
            )


def save_results(out_path: Path, records: list[dict], summary: dict, exceed: dict | None) -> None:
    suffix = out_path.suffix.lower()
    if suffix == ".json":
        payload = {
            "summary": summary,
            "exceed_ratios": exceed,
            "windows": records,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    elif suffix == ".csv":
        # 윈도우 단위 raw record를 CSV로 (집계는 stdout). pandas로 안정적 직렬화.
        pd.DataFrame.from_records(records).to_csv(out_path, index=False)
    else:
        raise ValueError(f"--out 확장자는 .csv 또는 .json 이어야 함. got {out_path.name}")
    print(f"\n[저장] {out_path}  (windows={len(records)})")


# ─── main ────────────────────────────────────────────────────────────────────

def collect_paths(
    data_dir: Path,
    activities: list[str] | None,
    environment: int | None = None,
) -> list[Path]:
    """data_dir의 SafeSignal CSV 중 activity/environment 필터를 통과한 경로.

    environment=None이면 환경 제한 없음 (기존 동작). 지정 시 해당 정수 환경만.
    """
    paths = sorted(data_dir.glob("*.csv"))
    valid: list[Path] = []
    act_filter = {a.upper() for a in activities} if activities else None
    for p in paths:
        try:
            meta = parse_safesignal_filename(p)
        except ValueError:
            continue
        if act_filter is not None and meta.activity.upper() not in act_filter:
            continue
        if environment is not None and meta.environment != environment:
            continue
        valid.append(p)
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="정식 전처리 기준 SDP z-score 전 energy 분석"
    )
    parser.add_argument("--dir", type=str, default="data/raw", help="CSV 폴더 (PROJECT_ROOT 기준)")
    parser.add_argument(
        "--activity", nargs="+", default=None,
        help="활동 코드 필터 (예: NO_MOTION WALK FALL_SIT_F). 없으면 전체.",
    )
    parser.add_argument(
        "--group-mode", choices=["activity", "class"], default="activity",
        help="집계 그룹 기준. activity=raw 코드, class=통합 클래스(FALL_*→fall).",
    )
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--max-gap-ms", type=float, default=100.0, help="resample gap 집계 임계(report-only)")
    parser.add_argument(
        "--limit-windows-per-file", type=int, default=None,
        help="파일당 처리할 최대 윈도우 수 (빠른 점검용).",
    )
    parser.add_argument("--max-files", type=int, default=None, help="처리할 최대 파일 수")
    parser.add_argument("--rpca-max-iter", type=int, default=DEFAULT_MAX_ITER,
                        help=f"RPCA ADMM 반복 (정식 기본 {DEFAULT_MAX_ITER})")
    parser.add_argument("--rpca-tol", type=float, default=None)
    parser.add_argument(
        "--workers", type=int, default=None,
        help="병렬 프로세스 수. None→cpu_count-1, 1→단일 프로세스.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=25,
        help=(
            "단일 프로세스(workers<=1)에서 N개 윈도우마다 진행률 출력. "
            "0이면 파일 완료 시에만 출력."
        ),
    )
    parser.add_argument("--out", type=str, default=None, help="결과 저장 경로 (.csv | .json)")
    args = parser.parse_args()

    data_dir = (PROJECT_ROOT / args.dir) if not Path(args.dir).is_absolute() else Path(args.dir)
    if not data_dir.exists():
        print(f"[ERROR] 폴더 없음: {data_dir}")
        sys.exit(1)

    paths = collect_paths(data_dir, args.activity)
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        print(f"[INFO] 대상 CSV 없음 (dir={data_dir}, activity={args.activity})")
        sys.exit(0)

    print(f"=== SDP energy 분석 ===")
    print(f"  대상 파일: {len(paths)}개  group_mode={args.group_mode}")
    print(f"  window={args.window_size} stride={args.stride} rpca_max_iter={args.rpca_max_iter}")
    if args.limit_windows_per_file is not None:
        print(f"  limit-windows-per-file={args.limit_windows_per_file}")

    workers = args.workers
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    progress_every = args.progress_every if workers <= 1 else 0
    if progress_every:
        print(f"  progress-every={progress_every} windows")

    work_args = [
        (
            str(p),
            args.group_mode,
            args.window_size,
            args.stride,
            args.max_gap_ms,
            args.limit_windows_per_file,
            args.rpca_max_iter,
            args.rpca_tol,
            progress_every,
        )
        for p in paths
    ]

    records: list[dict] = []
    done = 0
    if workers <= 1:
        for wa in work_args:
            name, recs, err = _process_file(wa)
            done += 1
            if err is not None:
                print(f"  [skip] {name}: {err}")
                continue
            records.extend(recs)
            print(f"  [{done}/{len(work_args)}] {name}: {len(recs)} windows")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_file, wa) for wa in work_args]
            for fut in as_completed(futures):
                name, recs, err = fut.result()
                done += 1
                if err is not None:
                    print(f"  [skip] {name}: {err}")
                    continue
                records.extend(recs)
                print(f"  [{done}/{len(futures)}] {name}: {len(recs)} windows")

    if not records:
        print("\n[INFO] 처리된 윈도우 없음")
        sys.exit(0)

    summary = aggregate(records)
    exceed = exceed_ratios(records, summary)
    print_report(summary, exceed)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        save_results(out_path, records, summary, exceed)

    print("\nDONE")


if __name__ == "__main__":
    main()
