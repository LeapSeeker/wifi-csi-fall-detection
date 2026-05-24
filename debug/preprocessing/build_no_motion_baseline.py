"""환경별 NO_MOTION calibration baseline JSON 생성 (분석용 summary).

목적
----
특정 환경(예: E2)의 NO_MOTION CSV를 정식 전처리 경로로 돌려 z-score 적용 *전*
SDP energy / sparse / raw-delta 분포와 수집 품질 지표를 요약한 baseline JSON을
``data/calibration/`` 에 저장한다. raw CSI CSV는 Git에 올리지 않고, 이 summary
JSON만 버전관리한다.

전처리 경로는 analyze_sdp_energy.py와 동일하다:
    load_safesignal_csv → resample_to_100hz → sliding_windows
      → rpca_sparse → stacked_doppler_profile → (z-score 직전) energy 계산

주의
----
이 파일은 추론 hard-skip / cooldown / margin / threshold 를 *결정하지 않는다*.
no_motion p99 와 fall 세션 energy p5 를 비교한 뒤에야 gate 적용 여부를 논의한다
(STATE D-020 / FALSE_POSITIVE_IMPROVEMENT_NOTES 참조). raw window record 전체는
JSON에 저장하지 않으며, 분포 summary만 보존한다.

실행 (PROJECT_ROOT = wifi-csi-fall-detection)
--------------------------------------------
    python debug/preprocessing/build_no_motion_baseline.py --env E2

    # 빠른 점검
    python debug/preprocessing/build_no_motion_baseline.py --env E2 \
        --limit-windows-per-file 1 --rpca-max-iter 1 --workers 1 --allow-few-files
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model.preprocessing.rpca import DEFAULT_MAX_ITER
from model.preprocessing.window import WINDOW_SIZE

# analyze_sdp_energy 의 정식 전처리 워커/상수를 재사용 (중복 구현 최소화)
from analyze_sdp_energy import (  # noqa: E402  (sys.path 주입 후 import)
    DEFAULT_STRIDE,
    _process_file,
    collect_paths,
)

SCHEMA_VERSION = 1

# baseline metrics 블록에 요약할 윈도우 단위 지표 (z-score 전)
BASELINE_METRICS = ["sdp_mean_abs", "sdp_std", "sparse_ratio", "raw_delta_mean"]

# baseline quality 블록에 요약할 파일 단위 지표 (None/NaN 은 percentile 제외)
QUALITY_METRICS = [
    "original_rate_hz",
    "max_gap_ms",
    "gap_count",
    "pair_dt_p50_us",
    "pair_dt_p95_us",
    "pair_dt_p99_us",
    "pair_dt_max_us",
]


def parse_environment(env_arg: str) -> tuple[str, int]:
    """'E2' 또는 '2' → ("E2", 2). 규칙에 안 맞으면 ValueError."""
    s = env_arg.strip().upper()
    digits = s[1:] if s.startswith("E") else s
    if not digits.isdigit():
        raise ValueError(f"--env 형식 오류: {env_arg!r} (예: E2 또는 2)")
    n = int(digits)
    return f"E{n}", n


def _percentile_summary(values: list[float]) -> dict | None:
    """None/NaN 제외 후 p50/p95/p99/min/max. 유효값 없으면 None."""
    arr = np.asarray(
        [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
        dtype=float,
    )
    if arr.size == 0:
        return None
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _dedupe_by_file(records: list[dict]) -> list[dict]:
    """윈도우 record에 broadcast된 파일 단위 ctx를 파일별 1건으로 축약."""
    seen: dict[str, dict] = {}
    for r in records:
        seen.setdefault(r["file"], r)
    return list(seen.values())


def build_baseline(
    records: list[dict],
    *,
    env_label: str,
    data_dir_rel: str,
    source_files: list[str],
    window_size: int,
    stride: int,
    rpca_max_iter: int,
    rpca_tol: float | None,
) -> dict:
    metrics_summary: dict[str, dict] = {}
    for metric in BASELINE_METRICS:
        s = _percentile_summary([r[metric] for r in records])
        if s is not None:
            metrics_summary[metric] = s

    per_file = _dedupe_by_file(records)
    quality_summary: dict[str, dict] = {}
    for metric in QUALITY_METRICS:
        s = _percentile_summary([f.get(metric) for f in per_file])
        if s is not None:
            quality_summary[metric] = s

    return {
        "schema_version": SCHEMA_VERSION,
        "environment": env_label,
        "activity": "NO_MOTION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "data_dir": data_dir_rel,
            "source_files": source_files,
            "n_files": len(source_files),
            "n_windows": len(records),
        },
        "preprocessing": {
            "window_size": window_size,
            "stride": stride,
            "rpca_max_iter": rpca_max_iter,
            "rpca_tol": rpca_tol,
            "resample_hz": 100,
        },
        "metrics": metrics_summary,
        "quality": quality_summary,
        "notes": [
            "Inference hard-skip threshold is not decided by this file.",
            "Compare no_motion p99 with fall p5 before applying any gate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="환경별 NO_MOTION calibration baseline JSON 생성 (summary only)"
    )
    parser.add_argument("--env", required=True, help="환경 라벨 (예: E2)")
    parser.add_argument("--dir", default="data/raw", help="CSV 폴더 (PROJECT_ROOT 기준)")
    parser.add_argument(
        "--out", default=None,
        help="저장 경로. 기본값은 data/calibration/{ENV}_no_motion_baseline.json",
    )
    parser.add_argument("--workers", type=int, default=1, help="병렬 프로세스 수 (기본 1)")
    parser.add_argument("--rpca-max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--rpca-tol", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit-windows-per-file", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None, help="처리할 최대 파일 수")
    parser.add_argument("--min-files", type=int, default=2, help="요구 최소 파일 수")
    parser.add_argument(
        "--allow-few-files", action="store_true",
        help="min-files 미달이어도 warning만 내고 진행",
    )
    args = parser.parse_args()

    try:
        env_label, env_int = parse_environment(args.env)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(2)

    data_dir = (PROJECT_ROOT / args.dir) if not Path(args.dir).is_absolute() else Path(args.dir)
    if not data_dir.exists():
        print(f"[ERROR] 폴더 없음: {data_dir}")
        sys.exit(1)

    paths = collect_paths(data_dir, ["NO_MOTION"], environment=env_int)
    if args.max_files is not None:
        paths = paths[: args.max_files]

    if not paths:
        print(f"[ERROR] 대상 CSV 없음 (env={env_label}, dir={data_dir}, activity=NO_MOTION)")
        sys.exit(1)

    if len(paths) < args.min_files:
        msg = (
            f"NO_MOTION 파일 {len(paths)}개 < min-files {args.min_files} "
            f"(env={env_label})"
        )
        if args.allow_few_files:
            print(f"[WARN] {msg} — --allow-few-files 지정, 진행")
        else:
            print(f"[ERROR] {msg} — --allow-few-files 없이는 중단")
            sys.exit(1)

    print(f"=== NO_MOTION baseline 생성 ({env_label}) ===")
    print(f"  대상 파일: {len(paths)}개")
    print(f"  window={WINDOW_SIZE} stride={DEFAULT_STRIDE} rpca_max_iter={args.rpca_max_iter}")
    if args.limit_windows_per_file is not None:
        print(f"  limit-windows-per-file={args.limit_windows_per_file}")

    workers = args.workers if args.workers and args.workers > 0 else 1
    progress_every = args.progress_every if workers <= 1 else 0

    work_args = [
        (
            str(p),
            "activity",            # group_mode (baseline에선 미사용, _process_file 시그니처 요구)
            WINDOW_SIZE,
            DEFAULT_STRIDE,
            100.0,                 # max_gap_ms — gap_count 집계용 (report-only)
            args.limit_windows_per_file,
            args.rpca_max_iter,
            args.rpca_tol,
            progress_every,
        )
        for p in paths
    ]

    records: list[dict] = []
    used_files: list[str] = []
    done = 0
    if workers <= 1:
        for wa in work_args:
            name, recs, err = _process_file(wa)
            done += 1
            if err is not None:
                print(f"  [skip] {name}: {err}")
                continue
            records.extend(recs)
            if recs:
                used_files.append(name)
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
                if recs:
                    used_files.append(name)
                print(f"  [{done}/{len(futures)}] {name}: {len(recs)} windows")

    if not records:
        print("\n[ERROR] 처리된 윈도우 없음")
        sys.exit(1)

    baseline = build_baseline(
        records,
        env_label=env_label,
        data_dir_rel=args.dir,
        source_files=sorted(used_files),
        window_size=WINDOW_SIZE,
        stride=DEFAULT_STRIDE,
        rpca_max_iter=args.rpca_max_iter,
        rpca_tol=args.rpca_tol,
    )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
    else:
        out_path = PROJECT_ROOT / "data" / "calibration" / f"{env_label}_no_motion_baseline.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"\n[저장] {out_path}\n"
        f"  files={baseline['source']['n_files']}  windows={baseline['source']['n_windows']}"
    )
    print("\nDONE")


if __name__ == "__main__":
    main()
