"""SafeSignal debug/analysis unified CLI.

긴 분석 명령을 매번 외우지 않도록 기존 debug 스크립트의 얇은 wrapper를 제공한다.
하위 스크립트는 항상 ``sys.executable``로 호출하여 현재 활성 Python/.venv를 유지한다.

하위 명령
---------
  quality           debug/data_collect/check_csv_quality.py 호출 (수집 CSV 품질)
  sdp-energy        debug/preprocessing/analyze_sdp_energy.py 호출 (활동별 energy)
  no-motion-energy  위와 동일하되 activity=NO_MOTION 고정 (--activity 미노출)

Examples
--------
CSV 품질 검사:
    python tools/safesignal_debug.py quality
    python tools/safesignal_debug.py quality --dir data/raw --fail-only

활동별 SDP z-score 전 energy 분석:
    python tools/safesignal_debug.py sdp-energy --activity NO_MOTION WALK SIT_STD

NO_MOTION 고정 energy 분석:
    python tools/safesignal_debug.py no-motion-energy --limit-windows-per-file 10 --progress-every 5
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
