"""NO_MOTION calibration baseline JSON 검증 / 요약 (read-only).

build_no_motion_baseline.py 가 생성한 baseline JSON을 *읽기만* 하여 schema와
핵심 값을 검증하고 사람이 읽기 쉬운 요약(또는 --json compact summary)을 출력한다.
이 스크립트는 절대 baseline 파일을 생성/수정/삭제하지 않는다.

입력 경로
--------
  --env E2  → data/calibration/E2_no_motion_baseline.json (자동)
  --path    → 해당 파일 직접 읽기 (--env보다 우선)
  둘 다 주면 --path 사용, 파일 내부 environment 가 --env 와 다르면 warning.

strict 모드(--strict)에서 non-zero exit 하는 조건
  - schema_version != 1
  - activity != NO_MOTION
  - source.n_files < 2
  - source.n_windows <= 0
  - preprocessing.rpca_max_iter < 100
  - 필수 metric 누락 (sdp_mean_abs / sdp_std / sparse_ratio / raw_delta_mean)

비-strict 모드에서는 위 문제를 warning 으로만 보고하고 exit 0 (단, 파일 없음 /
JSON 파싱 실패는 strict 여부와 무관하게 항상 non-zero exit).

실행 (PROJECT_ROOT = wifi-csi-fall-detection)
--------------------------------------------
    python debug/preprocessing/inspect_no_motion_baseline.py --env E2
    python debug/preprocessing/inspect_no_motion_baseline.py --env E2 --strict
    python debug/preprocessing/inspect_no_motion_baseline.py --path data/calibration/E2_no_motion_baseline.json
    python debug/preprocessing/inspect_no_motion_baseline.py --env E2 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_METRICS = ["sdp_mean_abs", "sdp_std", "sparse_ratio", "raw_delta_mean"]
# quality 블록 중 요약 출력 대상 (없으면 조용히 생략)
QUALITY_KEYS = ["original_rate_hz", "max_gap_ms", "gap_count", "pair_dt_p99_us"]

EXIT_OK = 0
EXIT_STRICT_FAIL = 1
EXIT_IO_FAIL = 2


def _resolve_path(args: argparse.Namespace) -> Path:
    """--path 우선, 없으면 --env 기준 기본 경로."""
    if args.path:
        p = Path(args.path)
    else:
        env_label = _normalize_env(args.env)
        p = PROJECT_ROOT / "data" / "calibration" / f"{env_label}_no_motion_baseline.json"
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _normalize_env(env_arg: str) -> str:
    s = env_arg.strip().upper()
    digits = s[1:] if s.startswith("E") else s
    if digits.isdigit():
        return f"E{int(digits)}"
    return s


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _fmt_metric(name: str, m: dict, keys: list[str]) -> str:
    parts = []
    for k in keys:
        v = m.get(k)
        parts.append(f"{k}={v:.5f}" if isinstance(v, (int, float)) else f"{k}=NA")
    return f"  {name:<16} " + " ".join(parts)


def validate(data: dict, env_arg: str | None, used_path_override: bool) -> tuple[list[str], list[str]]:
    """schema / 핵심 값 검증. returns (warnings, errors)."""
    warnings: list[str] = []
    errors: list[str] = []

    schema_version = data.get("schema_version")
    if schema_version != 1:
        errors.append(f"schema_version != 1 (got {schema_version!r})")

    activity = data.get("activity")
    if activity != "NO_MOTION":
        errors.append(f"activity != NO_MOTION (got {activity!r})")

    source = data.get("source") or {}
    n_files = source.get("n_files")
    if not isinstance(n_files, int) or n_files < 2:
        errors.append(f"source.n_files < 2 (got {n_files!r})")
    n_windows = source.get("n_windows")
    if not isinstance(n_windows, int) or n_windows <= 0:
        errors.append(f"source.n_windows <= 0 (got {n_windows!r})")

    preprocessing = data.get("preprocessing") or {}
    rpca_max_iter = preprocessing.get("rpca_max_iter")
    if not isinstance(rpca_max_iter, int) or rpca_max_iter < 100:
        errors.append(f"preprocessing.rpca_max_iter < 100 (got {rpca_max_iter!r})")

    metrics = data.get("metrics") or {}
    for name in REQUIRED_METRICS:
        if name not in metrics:
            errors.append(f"필수 metric 누락: metrics.{name}")

    # environment 불일치 (override 시에만 의미 있음): 항상 warning
    if used_path_override and env_arg:
        file_env = data.get("environment")
        want = _normalize_env(env_arg)
        if file_env is not None and str(file_env) != want:
            warnings.append(
                f"파일 environment({file_env!r}) != --env({want!r})"
            )

    return warnings, errors


def _core_metric_summary(data: dict) -> dict:
    """--json 출력용 핵심 metric summary (있는 것만)."""
    metrics = data.get("metrics") or {}
    out: dict[str, dict] = {}
    for name in REQUIRED_METRICS:
        m = metrics.get(name)
        if isinstance(m, dict):
            out[name] = {k: m.get(k) for k in ("p50", "p95", "p99", "min", "max")}
    return out


def print_human(path: Path, data: dict, warnings: list[str]) -> None:
    source = data.get("source") or {}
    preprocessing = data.get("preprocessing") or {}
    metrics = data.get("metrics") or {}
    quality = data.get("quality") or {}

    print("=== NO_MOTION baseline summary ===")
    print(f"file: {_rel(path)}")
    print(f"environment: {data.get('environment')}")
    print(f"activity: {data.get('activity')}")
    print(f"created_at: {data.get('created_at')}")
    print(f"files/windows: {source.get('n_files')} files / {source.get('n_windows')} windows")
    print(f"rpca_max_iter: {preprocessing.get('rpca_max_iter')}")

    source_files = source.get("source_files") or []
    if source_files:
        print("source_files:")
        for f in source_files:
            print(f"  - {f}")

    print("\nmetrics:")
    for name in REQUIRED_METRICS:
        m = metrics.get(name)
        if isinstance(m, dict):
            print(_fmt_metric(name, m, ["p50", "p95", "p99", "min", "max"]))
        else:
            print(f"  {name:<16} (누락)")

    if quality:
        print("\nquality:")
        for name in QUALITY_KEYS:
            m = quality.get(name)
            if isinstance(m, dict):
                print(_fmt_metric(name, m, ["p50", "p95", "p99", "max"]))

    print()
    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("warnings: none")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NO_MOTION baseline JSON 검증/요약 (read-only)"
    )
    parser.add_argument("--env", default=None, help="환경 라벨 (예: E2)")
    parser.add_argument("--path", default=None, help="baseline JSON 직접 지정 (--env보다 우선)")
    parser.add_argument("--strict", action="store_true", help="검증 실패 시 non-zero exit")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="compact JSON summary 출력")
    args = parser.parse_args()

    if not args.env and not args.path:
        print("[ERROR] --env 또는 --path 중 하나는 필요합니다.")
        sys.exit(EXIT_IO_FAIL)

    path = _resolve_path(args)

    if not path.exists():
        print(f"[ERROR] baseline 파일 없음: {_rel(path)}")
        sys.exit(EXIT_IO_FAIL)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        print(f"[ERROR] baseline JSON 파싱 실패: {_rel(path)} — {e}")
        sys.exit(EXIT_IO_FAIL)

    if not isinstance(data, dict):
        print(f"[ERROR] baseline JSON 최상위가 object 가 아님: {_rel(path)}")
        sys.exit(EXIT_IO_FAIL)

    used_path_override = bool(args.path)
    warnings, errors = validate(data, args.env, used_path_override)

    # strict 모드: error를 fail 사유로, 비-strict: error도 warning으로 강등하여 보고
    fail = args.strict and bool(errors)
    if not args.strict:
        warnings = warnings + errors
        errors_for_output: list[str] = []
    else:
        errors_for_output = errors

    if args.as_json:
        out = {
            "ok": not fail,
            "file": _rel(path),
            "environment": data.get("environment"),
            "activity": data.get("activity"),
            "n_files": (data.get("source") or {}).get("n_files"),
            "n_windows": (data.get("source") or {}).get("n_windows"),
            "rpca_max_iter": (data.get("preprocessing") or {}).get("rpca_max_iter"),
            "metrics": _core_metric_summary(data),
            "warnings": warnings,
            "errors": errors_for_output,
        }
        print(json.dumps(out, ensure_ascii=False))
    else:
        print_human(path, data, warnings)
        if args.strict and errors_for_output:
            print("\nerrors:")
            for e in errors_for_output:
                print(f"  - {e}")

    sys.exit(EXIT_STRICT_FAIL if fail else EXIT_OK)


if __name__ == "__main__":
    main()
