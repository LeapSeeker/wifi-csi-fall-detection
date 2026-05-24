"""NO_MOTION baseline 대비 대상 활동/FALL의 z-score 전 SDP energy 분리도 비교 (read-only).

목적
----
build_no_motion_baseline.py 가 만든 NO_MOTION baseline JSON의 분포(p95/p99)와,
대상 CSV(기본 fall class)의 z-score 적용 *전* SDP energy 분포(p1/p5/...)를 비교하여
"no_motion p99"와 "target p5"가 얼마나 분리되는지 확인한다.

이 스크립트는 추론 threshold / hard-skip / margin / cooldown / event detector를
*결정하거나 구현하지 않는다*. 단지 분리 가능성(separation)을 read-only로 보고한다.
(STATE D-020 / FALSE_POSITIVE_IMPROVEMENT_NOTES 7.5 / §8.5 참조)

전처리 경로는 analyze_sdp_energy.py / build_no_motion_baseline.py와 동일하다:
    load_safesignal_csv → resample_to_100hz → sliding_windows
      → rpca_sparse → stacked_doppler_profile → (z-score 직전) energy 계산

raw window record 전체는 저장하지 않는다. --out 지정 시 summary JSON만 저장한다.

입력 / 동작
----------
  baseline:
    --env E2          → data/calibration/E2_no_motion_baseline.json (기본 경로)
    --baseline PATH   → 해당 파일 직접 읽기 (--env 보다 우선)
    파일 없음 / JSON 파싱 실패 → non-zero exit.
    schema_version != 1 또는 activity != NO_MOTION → warning (중단하지 않음).

  대상 CSV 선택:
    --dir data/raw    기본 폴더
    --target-env      미지정 시 baseline environment 와 동일
    --target-class    기본 fall — activity_to_class() 기준 class 필터
    --activity        지정 시 raw activity 직접 선택 (--target-class 보다 우선, warning)
    대상 파일 없음 → 명확한 메시지 + non-zero exit.

실행 (PROJECT_ROOT = wifi-csi-fall-detection)
--------------------------------------------
    python debug/preprocessing/compare_sdp_energy_to_baseline.py --env E2
    python debug/preprocessing/compare_sdp_energy_to_baseline.py --env E2 --target-env E4
    python debug/preprocessing/compare_sdp_energy_to_baseline.py \
        --baseline data/calibration/E2_no_motion_baseline.json --target-class fall
    python debug/preprocessing/compare_sdp_energy_to_baseline.py --env E2 --activity WALK

    # 빠른 점검 (정상 경로)
    python debug/preprocessing/compare_sdp_energy_to_baseline.py \
        --env E2 --target-env E4 --activity WALK \
        --limit-windows-per-file 1 --rpca-max-iter 1 --workers 1

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

from model.preprocessing.loader import parse_safesignal_filename
from model.preprocessing.rpca import DEFAULT_MAX_ITER
from model.preprocessing.window import WINDOW_SIZE

# analyze_sdp_energy 의 정식 전처리 워커/상수를 재사용 (중복 구현 최소화)
from analyze_sdp_energy import (  # noqa: E402  (sys.path 주입 후 import)
    DEFAULT_STRIDE,
    _process_file,
    activity_to_class,
    collect_paths,
)

SCHEMA_VERSION = 1

# baseline 대비 비교할 윈도우 단위 지표 (z-score 전)
COMPARE_METRICS = ["sdp_mean_abs", "sdp_std", "sparse_ratio", "raw_delta_mean"]

EXIT_OK = 0
EXIT_NO_TARGET = 1
EXIT_IO_FAIL = 2


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _env_label_and_int(env_arg: str) -> tuple[str, int]:
    """'E2' 또는 '2' → ('E2', 2). 규칙에 안 맞으면 ValueError."""
    s = str(env_arg).strip().upper()
    digits = s[1:] if s.startswith("E") else s
    if not digits.isdigit():
        raise ValueError(f"환경 라벨 형식 오류: {env_arg!r} (예: E2 또는 2)")
    n = int(digits)
    return f"E{n}", n


# ─── baseline 로드 ───────────────────────────────────────────────────────────

def resolve_baseline_path(args: argparse.Namespace) -> Path:
    """--baseline 우선, 없으면 --env 기준 기본 경로. 둘 다 없으면 None 반환."""
    if args.baseline:
        p = Path(args.baseline)
    elif args.env:
        env_label, _ = _env_label_and_int(args.env)
        p = PROJECT_ROOT / "data" / "calibration" / f"{env_label}_no_motion_baseline.json"
    else:
        return None  # type: ignore[return-value]
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_baseline(path: Path) -> tuple[dict, list[str]]:
    """baseline JSON 로드 + schema warning 수집. IO/parse 실패는 호출부에서 exit."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("baseline JSON 최상위가 object 가 아님")

    warnings: list[str] = []
    if data.get("schema_version") != 1:
        warnings.append(f"baseline schema_version != 1 (got {data.get('schema_version')!r})")
    if data.get("activity") != "NO_MOTION":
        warnings.append(f"baseline activity != NO_MOTION (got {data.get('activity')!r})")
    return data, warnings


# ─── 대상 CSV 선택 ───────────────────────────────────────────────────────────

def select_target_paths(
    data_dir: Path,
    *,
    target_env_int: int,
    activities: list[str] | None,
    target_class: str | None,
) -> list[Path]:
    """activity 지정 시 raw activity 필터, 아니면 activity_to_class 기준 class 필터."""
    if activities:
        return collect_paths(data_dir, activities, environment=target_env_int)

    # class 필터: 해당 환경의 모든 CSV를 모은 뒤 activity_to_class 로 거른다.
    env_paths = collect_paths(data_dir, None, environment=target_env_int)
    out: list[Path] = []
    for p in env_paths:
        try:
            meta = parse_safesignal_filename(p)
        except ValueError:
            continue
        if activity_to_class(meta.activity) == target_class:
            out.append(p)
    return out


# ─── 비교 계산 ───────────────────────────────────────────────────────────────

def _target_percentiles(values: np.ndarray) -> dict:
    return {
        "target_min": float(values.min()),
        "target_p1": float(np.percentile(values, 1)),
        "target_p5": float(np.percentile(values, 5)),
        "target_p50": float(np.percentile(values, 50)),
        "target_p95": float(np.percentile(values, 95)),
        "target_p99": float(np.percentile(values, 99)),
    }


def _baseline_p95_p99(baseline: dict, metric: str) -> tuple[float | None, float | None]:
    m = (baseline.get("metrics") or {}).get(metric)
    if not isinstance(m, dict):
        return None, None
    p95 = m.get("p95")
    p99 = m.get("p99")
    return (
        float(p95) if isinstance(p95, (int, float)) else None,
        float(p99) if isinstance(p99, (int, float)) else None,
    )


def build_comparison(
    records: list[dict], baseline: dict
) -> tuple[dict, list[str]]:
    """metric → 비교 dict. baseline metric 누락 시 warning."""
    comparison: dict[str, dict] = {}
    warnings: list[str] = []

    for metric in COMPARE_METRICS:
        vals = np.asarray([r[metric] for r in records], dtype=float)
        base_p95, base_p99 = _baseline_p95_p99(baseline, metric)
        if base_p95 is None and base_p99 is None:
            warnings.append(f"baseline metrics.{metric} 누락 — 비교는 target 분포만 표시")

        tp = _target_percentiles(vals)
        target_p5 = tp["target_p5"]

        if base_p99 is None:
            separation = None
            ratio = None
            ratio_le = None
        else:
            separation = float(target_p5 - base_p99)
            ratio = float(target_p5 / base_p99) if base_p99 != 0 else None
            ratio_le = float((vals <= base_p99).mean())

        comparison[metric] = {
            "baseline_p95": base_p95,
            "baseline_p99": base_p99,
            **tp,
            "separation_p5_minus_baseline_p99": separation,
            "ratio_p5_over_baseline_p99": ratio,
            "target_ratio_le_baseline_p99": ratio_le,
        }
    return comparison, warnings


# ─── 출력 ────────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    return f"{v:.5f}" if isinstance(v, (int, float)) else "NA"


def print_report(
    *,
    baseline_path: Path,
    baseline_env,
    target_env_label: str,
    target_selector: dict,
    n_files: int,
    n_windows: int,
    comparison: dict,
    warnings: list[str],
) -> None:
    if target_selector["activity"]:
        sel = f"activity={' '.join(target_selector['activity'])}"
    else:
        sel = f"class={target_selector['class']}"

    print("=== SDP energy baseline comparison ===")
    print(f"baseline: {_rel(baseline_path)}")
    print(f"baseline environment: {baseline_env}")
    print(f"target environment: {target_env_label}")
    print(f"target selector: {sel}")
    print(f"files/windows: {n_files} files / {n_windows} windows")

    for metric in COMPARE_METRICS:
        c = comparison[metric]
        print(f"\n[{metric}]")
        print(f"  no_motion p95={_fmt(c['baseline_p95'])}")
        print(f"  no_motion p99={_fmt(c['baseline_p99'])}")
        print(f"  target min={_fmt(c['target_min'])}")
        print(f"  target p1={_fmt(c['target_p1'])}")
        print(f"  target p5={_fmt(c['target_p5'])}")
        print(f"  target p50={_fmt(c['target_p50'])}")
        print(f"  target p95={_fmt(c['target_p95'])}")
        print(f"  target p99={_fmt(c['target_p99'])}")
        print(f"  separation_p5_minus_nm_p99={_fmt(c['separation_p5_minus_baseline_p99'])}")
        print(f"  ratio_p5_over_nm_p99={_fmt(c['ratio_p5_over_baseline_p99'])}")
        print(f"  target_ratio_le_nm_p99={_fmt(c['target_ratio_le_baseline_p99'])}")

        sep = c["separation_p5_minus_baseline_p99"]
        print("  interpretation:")
        if sep is None:
            print("    - baseline p99 없음: 분리 판정 불가 (baseline metric 누락)")
        elif sep > 0:
            print("    - target p5 > no_motion p99: separated in sampled data")
        else:
            print("    - overlap exists; do not use this metric as hard gate")

    print()
    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("warnings: none")


def build_summary_json(
    *,
    baseline_path: Path,
    baseline_env,
    target_env_label: str,
    target_selector: dict,
    data_dir_rel: str,
    source_files: list[str],
    n_windows: int,
    comparison: dict,
    warnings: list[str],
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_file": _rel(baseline_path),
        "baseline_environment": baseline_env,
        "target_environment": target_env_label,
        "target_selector": target_selector,
        "source": {
            "data_dir": data_dir_rel,
            "source_files": source_files,
            "n_files": len(source_files),
            "n_windows": n_windows,
        },
        "comparison": comparison,
        "warnings": warnings,
    }


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NO_MOTION baseline 대비 SDP z-score 전 energy 분리도 비교 (read-only)"
    )
    parser.add_argument("--env", default=None, help="baseline 환경 라벨 (예: E2) — 기본 baseline 경로 결정")
    parser.add_argument("--baseline", default=None, help="baseline JSON 직접 지정 (--env 보다 우선)")
    parser.add_argument("--dir", default="data/raw", help="CSV 폴더 (PROJECT_ROOT 기준)")
    parser.add_argument("--target-env", default=None, help="대상 환경 (미지정 시 baseline environment)")
    parser.add_argument("--target-class", default=None, help="대상 class (기본 fall, activity_to_class 기준)")
    parser.add_argument("--activity", nargs="+", default=None, help="대상 raw activity (지정 시 --target-class 보다 우선)")
    parser.add_argument("--workers", type=int, default=1, help="병렬 프로세스 수 (기본 1)")
    parser.add_argument("--rpca-max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--rpca-tol", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit-windows-per-file", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None, help="처리할 최대 파일 수")
    parser.add_argument("--out", default=None, help="summary JSON 저장 경로 (raw window record는 저장 안 함)")
    args = parser.parse_args()

    warnings: list[str] = []

    # 1) baseline 경로 결정 + 로드
    baseline_path = resolve_baseline_path(args)
    if baseline_path is None:
        print("[ERROR] --env 또는 --baseline 중 하나는 필요합니다.")
        sys.exit(EXIT_IO_FAIL)
    if not baseline_path.exists():
        print(f"[ERROR] baseline 파일 없음: {_rel(baseline_path)}")
        sys.exit(EXIT_IO_FAIL)
    try:
        baseline, base_warnings = load_baseline(baseline_path)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError) as e:
        print(f"[ERROR] baseline JSON 로드 실패: {_rel(baseline_path)} — {e}")
        sys.exit(EXIT_IO_FAIL)
    warnings.extend(base_warnings)

    baseline_env = baseline.get("environment")

    # 2) 대상 환경 결정
    if args.target_env:
        try:
            target_env_label, target_env_int = _env_label_and_int(args.target_env)
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(EXIT_IO_FAIL)
    else:
        try:
            target_env_label, target_env_int = _env_label_and_int(baseline_env)
        except (ValueError, TypeError):
            print(
                f"[ERROR] baseline environment({baseline_env!r})를 해석할 수 없습니다. "
                "--target-env 를 명시하세요."
            )
            sys.exit(EXIT_IO_FAIL)

    # 3) 대상 selector 결정 (--activity 우선)
    if args.activity:
        if args.target_class is not None:
            warnings.append("--activity 와 --target-class 동시 지정 — --activity 우선 사용")
        target_selector = {"class": None, "activity": [a.upper() for a in args.activity]}
        activities = args.activity
        target_class = None
    else:
        target_class = args.target_class or "fall"
        target_selector = {"class": target_class, "activity": None}
        activities = None

    data_dir = (PROJECT_ROOT / args.dir) if not Path(args.dir).is_absolute() else Path(args.dir)
    if not data_dir.exists():
        print(f"[ERROR] 폴더 없음: {data_dir}")
        sys.exit(EXIT_IO_FAIL)

    # 4) 대상 CSV 선택
    paths = select_target_paths(
        data_dir,
        target_env_int=target_env_int,
        activities=activities,
        target_class=target_class,
    )
    if args.max_files is not None:
        paths = paths[: args.max_files]

    if not paths:
        sel = (
            f"activity={' '.join(activities)}" if activities else f"class={target_class}"
        )
        print(
            f"[ERROR] 대상 CSV 없음 (env={target_env_label}, {sel}, dir={_rel(data_dir)}). "
            "로컬에 FALL CSV가 아직 없을 수 있습니다."
        )
        sys.exit(EXIT_NO_TARGET)

    print("=== SDP energy baseline comparison ===")
    print(f"baseline: {_rel(baseline_path)}  (env={baseline_env})")
    print(f"대상 파일: {len(paths)}개  target_env={target_env_label}")
    print(f"  window={WINDOW_SIZE} stride={DEFAULT_STRIDE} rpca_max_iter={args.rpca_max_iter}")
    if args.limit_windows_per_file is not None:
        print(f"  limit-windows-per-file={args.limit_windows_per_file}")

    workers = args.workers if args.workers and args.workers > 0 else 1
    progress_every = args.progress_every if workers <= 1 else 0

    work_args = [
        (
            str(p),
            "activity",            # group_mode (여기선 미사용, _process_file 시그니처 요구)
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

    # 5) 정식 전처리 경로로 처리
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
        print("\n[ERROR] 처리된 윈도우 없음 (대상 파일은 있으나 윈도우 생성 실패)")
        sys.exit(EXIT_NO_TARGET)

    # 6) 비교 계산 + 출력
    comparison, cmp_warnings = build_comparison(records, baseline)
    warnings.extend(cmp_warnings)

    print()
    print_report(
        baseline_path=baseline_path,
        baseline_env=baseline_env,
        target_env_label=target_env_label,
        target_selector=target_selector,
        n_files=len(used_files),
        n_windows=len(records),
        comparison=comparison,
        warnings=warnings,
    )

    # 7) summary JSON 저장 (raw window record는 저장하지 않음)
    if args.out:
        summary = build_summary_json(
            baseline_path=baseline_path,
            baseline_env=baseline_env,
            target_env_label=target_env_label,
            target_selector=target_selector,
            data_dir_rel=args.dir,
            source_files=sorted(used_files),
            n_windows=len(records),
            comparison=comparison,
            warnings=warnings,
        )
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[저장] {_rel(out_path)}  (summary only, windows={len(records)})")

    print("\nDONE")


if __name__ == "__main__":
    main()
