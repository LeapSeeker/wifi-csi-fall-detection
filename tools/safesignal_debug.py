"""SafeSignal debug/analysis unified CLI.

긴 분석 명령을 매번 외우지 않도록 기존 debug 스크립트의 얇은 wrapper를 제공한다.
하위 스크립트는 항상 ``sys.executable``로 호출하여 현재 활성 Python/.venv를 유지한다.

하위 명령
---------
  quality           debug/data_collect/check_csv_quality.py 호출 (수집 CSV 품질)
  sdp-energy        debug/preprocessing/analyze_sdp_energy.py 호출 (활동별 energy)
  no-motion-energy  위와 동일하되 activity=NO_MOTION 고정 (--activity 미노출)
  calibrate         debug/preprocessing/build_no_motion_baseline.py 호출
                    (환경별 NO_MOTION baseline JSON summary 생성)
  baseline-summary  debug/preprocessing/inspect_no_motion_baseline.py 호출
                    (baseline JSON 검증/요약, read-only)
  compare-energy    debug/preprocessing/compare_sdp_energy_to_baseline.py 호출
                    (baseline 대비 대상 활동/FALL energy 분리도 비교, read-only)

Examples
--------
CSV 품질 검사:
    python tools/safesignal_debug.py quality
    python tools/safesignal_debug.py quality --dir data/raw --fail-only

활동별 SDP z-score 전 energy 분석:
    python tools/safesignal_debug.py sdp-energy --activity NO_MOTION WALK SIT_STD

NO_MOTION 고정 energy 분석:
    python tools/safesignal_debug.py no-motion-energy --limit-windows-per-file 10 --progress-every 5

환경별 NO_MOTION baseline 생성:
    python tools/safesignal_debug.py calibrate --env E2
    python tools/safesignal_debug.py calibrate --env E2 --workers 1 --progress-every 10

baseline JSON 검증/요약:
    python tools/safesignal_debug.py baseline-summary --env E2
    python tools/safesignal_debug.py baseline-summary --env E2 --strict
    python tools/safesignal_debug.py baseline-summary --env E2 --json

baseline 대비 SDP energy 분리도 비교 (read-only):
    python tools/safesignal_debug.py compare-energy --env E2
    python tools/safesignal_debug.py compare-energy --env E2 --target-env E4
    python tools/safesignal_debug.py compare-energy --baseline data/calibration/E2_no_motion_baseline.json --target-class fall
    python tools/safesignal_debug.py compare-energy --env E2 --activity WALK
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: Path, args: list[str]) -> int:
    cmd = [sys.executable, str(script), *args]
    print("[run]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def _add_sdp_args(parser: argparse.ArgumentParser, include_activity: bool = True) -> None:
    parser.add_argument("--dir", default="data/raw", help="CSV 폴더")
    if include_activity:
        parser.add_argument("--activity", nargs="+", default=None, help="활동 코드 필터")
    parser.add_argument(
        "--group-mode",
        choices=["activity", "class"],
        default="activity",
        help="집계 기준",
    )
    parser.add_argument("--workers", type=int, default=1, help="병렬 프로세스 수")
    parser.add_argument("--rpca-max-iter", type=int, default=200)
    parser.add_argument("--rpca-tol", type=float, default=None)
    parser.add_argument("--limit-windows-per-file", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--out", default=None, help="결과 저장 경로 (.csv | .json)")


def _sdp_forward_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = [
        "--dir",
        args.dir,
        "--group-mode",
        args.group_mode,
        "--workers",
        str(args.workers),
        "--rpca-max-iter",
        str(args.rpca_max_iter),
        "--progress-every",
        str(args.progress_every),
    ]
    if getattr(args, "activity", None):
        out.extend(["--activity", *args.activity])
    if args.rpca_tol is not None:
        out.extend(["--rpca-tol", str(args.rpca_tol)])
    if args.limit_windows_per_file is not None:
        out.extend(["--limit-windows-per-file", str(args.limit_windows_per_file)])
    if args.max_files is not None:
        out.extend(["--max-files", str(args.max_files)])
    if args.out:
        out.extend(["--out", args.out])
    return out


def cmd_quality(args: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "debug" / "data_collect" / "check_csv_quality.py"
    forwarded = ["--dir", args.dir]
    if args.fail_only:
        forwarded.append("--fail_only")
    return _run_script(script, forwarded)


def cmd_sdp_energy(args: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "debug" / "preprocessing" / "analyze_sdp_energy.py"
    return _run_script(script, _sdp_forward_args(args))


def cmd_no_motion_energy(args: argparse.Namespace) -> int:
    args.activity = ["NO_MOTION"]
    script = PROJECT_ROOT / "debug" / "preprocessing" / "analyze_sdp_energy.py"
    return _run_script(script, _sdp_forward_args(args))


def cmd_calibrate(args: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "debug" / "preprocessing" / "build_no_motion_baseline.py"
    forwarded: list[str] = [
        "--env",
        args.env,
        "--dir",
        args.dir,
        "--workers",
        str(args.workers),
        "--rpca-max-iter",
        str(args.rpca_max_iter),
        "--progress-every",
        str(args.progress_every),
        "--min-files",
        str(args.min_files),
    ]
    if args.out:
        forwarded.extend(["--out", args.out])
    if args.rpca_tol is not None:
        forwarded.extend(["--rpca-tol", str(args.rpca_tol)])
    if args.limit_windows_per_file is not None:
        forwarded.extend(["--limit-windows-per-file", str(args.limit_windows_per_file)])
    if args.max_files is not None:
        forwarded.extend(["--max-files", str(args.max_files)])
    if args.allow_few_files:
        forwarded.append("--allow-few-files")
    return _run_script(script, forwarded)


def cmd_baseline_summary(args: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "debug" / "preprocessing" / "inspect_no_motion_baseline.py"
    forwarded: list[str] = []
    if args.env:
        forwarded.extend(["--env", args.env])
    if args.path:
        forwarded.extend(["--path", args.path])
    if args.strict:
        forwarded.append("--strict")
    if args.as_json:
        forwarded.append("--json")
    return _run_script(script, forwarded)


def cmd_compare_energy(args: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "debug" / "preprocessing" / "compare_sdp_energy_to_baseline.py"
    forwarded: list[str] = ["--dir", args.dir]
    if args.env:
        forwarded.extend(["--env", args.env])
    if args.baseline:
        forwarded.extend(["--baseline", args.baseline])
    if args.target_env:
        forwarded.extend(["--target-env", args.target_env])
    if args.target_class is not None:
        forwarded.extend(["--target-class", args.target_class])
    if args.activity:
        forwarded.extend(["--activity", *args.activity])
    forwarded.extend([
        "--workers", str(args.workers),
        "--rpca-max-iter", str(args.rpca_max_iter),
        "--progress-every", str(args.progress_every),
    ])
    if args.rpca_tol is not None:
        forwarded.extend(["--rpca-tol", str(args.rpca_tol)])
    if args.limit_windows_per_file is not None:
        forwarded.extend(["--limit-windows-per-file", str(args.limit_windows_per_file)])
    if args.max_files is not None:
        forwarded.extend(["--max-files", str(args.max_files)])
    if args.out:
        forwarded.extend(["--out", args.out])
    return _run_script(script, forwarded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SafeSignal debug/analysis runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_quality = sub.add_parser("quality", help="수집 CSV 품질 검사")
    p_quality.add_argument("--dir", default="data/raw", help="CSV 폴더")
    p_quality.add_argument("--fail-only", action="store_true", help="기준 미달만 출력")
    p_quality.set_defaults(func=cmd_quality)

    p_sdp = sub.add_parser("sdp-energy", help="활동별 SDP z-score 전 energy 분석")
    _add_sdp_args(p_sdp)
    p_sdp.set_defaults(func=cmd_sdp_energy)

    p_nm = sub.add_parser("no-motion-energy", help="NO_MOTION 고정 SDP energy 분석")
    _add_sdp_args(p_nm, include_activity=False)
    p_nm.set_defaults(func=cmd_no_motion_energy)

    p_cal = sub.add_parser("calibrate", help="환경별 NO_MOTION baseline JSON 생성")
    p_cal.add_argument("--env", required=True, help="환경 라벨 (예: E2)")
    p_cal.add_argument("--dir", default="data/raw", help="CSV 폴더")
    p_cal.add_argument(
        "--out", default=None,
        help="저장 경로 (기본 data/calibration/{ENV}_no_motion_baseline.json)",
    )
    p_cal.add_argument("--workers", type=int, default=1, help="병렬 프로세스 수")
    p_cal.add_argument("--rpca-max-iter", type=int, default=200)
    p_cal.add_argument("--rpca-tol", type=float, default=None)
    p_cal.add_argument("--limit-windows-per-file", type=int, default=None)
    p_cal.add_argument("--progress-every", type=int, default=25)
    p_cal.add_argument("--max-files", type=int, default=None)
    p_cal.add_argument("--min-files", type=int, default=2, help="요구 최소 파일 수")
    p_cal.add_argument(
        "--allow-few-files", action="store_true",
        help="min-files 미달이어도 warning만 내고 진행",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    p_sum = sub.add_parser("baseline-summary", help="NO_MOTION baseline JSON 검증/요약 (read-only)")
    p_sum.add_argument("--env", default=None, help="환경 라벨 (예: E2)")
    p_sum.add_argument("--path", default=None, help="baseline JSON 직접 지정 (--env보다 우선)")
    p_sum.add_argument("--strict", action="store_true", help="검증 실패 시 non-zero exit")
    p_sum.add_argument("--json", action="store_true", dest="as_json", help="compact JSON summary 출력")
    p_sum.set_defaults(func=cmd_baseline_summary)

    p_cmp = sub.add_parser(
        "compare-energy",
        help="baseline 대비 대상 활동/FALL SDP energy 분리도 비교 (read-only)",
    )
    p_cmp.add_argument("--env", default=None, help="baseline 환경 라벨 (예: E2)")
    p_cmp.add_argument("--baseline", default=None, help="baseline JSON 직접 지정 (--env 보다 우선)")
    p_cmp.add_argument("--dir", default="data/raw", help="CSV 폴더")
    p_cmp.add_argument("--target-env", default=None, help="대상 환경 (미지정 시 baseline environment)")
    p_cmp.add_argument("--target-class", default=None, help="대상 class (기본 fall)")
    p_cmp.add_argument("--activity", nargs="+", default=None, help="대상 raw activity (--target-class 보다 우선)")
    p_cmp.add_argument("--workers", type=int, default=1, help="병렬 프로세스 수")
    p_cmp.add_argument("--rpca-max-iter", type=int, default=200)
    p_cmp.add_argument("--rpca-tol", type=float, default=None)
    p_cmp.add_argument("--limit-windows-per-file", type=int, default=None)
    p_cmp.add_argument("--progress-every", type=int, default=25)
    p_cmp.add_argument("--max-files", type=int, default=None)
    p_cmp.add_argument("--out", default=None, help="summary JSON 저장 경로 (raw record 미저장)")
    p_cmp.set_defaults(func=cmd_compare_energy)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
