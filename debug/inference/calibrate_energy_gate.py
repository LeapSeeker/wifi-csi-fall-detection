"""SDP energy gate 캘리브레이션 — 정적/라이브 CSV에서 energy 분포 → gate threshold 후보.

목적
----
오늘 무인 정적 측정과 데모 당일 현장 측정, 두 번 같은 절차로 실행해 energy gate
threshold 후보를 산출한다. 모델 추론은 하지 않고 energy 만 계산한다.

energy 계산은 predictor.py 의 gate 와 **동일한 단일 helper**
(`server.inference.energy.compute_window_energy`)를 호출한다. 따라서 여기서 고른
threshold 가 실제 gate 에서 같은 값으로 동작한다(지시서 보강: helper 단일화).

윈도잉도 서버 실시간 경로와 동일하게 SlidingWindowBuffer(100Hz resample / stride
100 / window 300)를 쓴다.

출력
----
  - selected metric 의 p50/p90/p95/p99/max
  - 전체 metric 분포(참고)
  - recommended threshold = selected_metric p99 * multiplier (기본 1.2)
  - --fall-patterns 지정 시 fall energy p5 (threshold 가 fall p5 보다 충분히 낮은지 확인용)
  - PowerShell env 설정 예시

주의
----
- 전체 calibration 은 RPCA 로 느리다. limit 옵션 필수.
- 오늘 무인 정적 측정에는 fall 세션이 없다. fall p5 는 기존 수집 fall CSV
  (예: data/raw 의 *FALL*.csv)에서 가져온다. fall 기준값은 한 번 계산해두면
  오늘/당일 공통 재사용 가능(임계만 환경별로 바뀜).

예시
----
    python debug/inference/calibrate_energy_gate.py --patterns "*NO_MOTION*.csv" \
        --metric sdp_mean_abs --multiplier 1.2

    # fall p5 와 함께 (정적 + 기존 fall 비교)
    python debug/inference/calibrate_energy_gate.py --patterns "*E3*NO_MOTION*.csv" \
        --fall-patterns "*FALL*.csv" --limit-files 5 --limit-windows-per-file 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "server") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "server"))

from model.preprocessing.loader import load_safesignal_csv  # noqa: E402
from server.inference.buffer import SlidingWindowBuffer  # noqa: E402
from server.inference.energy import ENERGY_METRICS, compute_window_energy  # noqa: E402

# predictor 와 동일하게 정적 gate 후보로 흔히 쓰는 정적 활동 패턴
DEFAULT_PATTERNS = ("*NO_MOTION*.csv", "*STAND*.csv", "*LIE*.csv")


def iter_paths(root: Path, patterns: list[str], limit_files: int | None) -> list[Path]:
    seen: dict[str, Path] = {}
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                seen[str(p.resolve())] = p
    paths = list(seen.values())
    if limit_files is not None:
        paths = paths[:limit_files]
    return paths


def collect_energy(
    paths: list[Path],
    limit_windows: int | None,
) -> dict[str, list[float]]:
    """파일들을 서버 윈도잉(SlidingWindowBuffer)으로 돌려 metric 별 값 리스트 수집."""
    values: dict[str, list[float]] = {m: [] for m in ENERGY_METRICS}
    for path in paths:
        try:
            raw = load_safesignal_csv(path, rx="both")
        except Exception as e:  # noqa: BLE001 — 파일 1개 실패가 전체를 막지 않게
            print(f"  [skip] {path.name}: {e!r}")
            continue
        amp = raw.amplitude
        ts = raw.timestamps_us
        if amp.ndim != 2 or amp.shape[1] != 104:
            print(f"  [skip] {path.name}: expected (N,104), got {amp.shape}")
            continue

        buf = SlidingWindowBuffer()
        n_win = 0
        for i in range(amp.shape[0]):
            triggered = buf.add(
                amp[i, :52],
                amp[i, 52:],
                timestamp_us=int(ts[i]) if ts is not None and len(ts) > i else 0,
            )
            if not triggered:
                continue
            _sdp, metrics = compute_window_energy(buf.get_window())
            for m in ENERGY_METRICS:
                values[m].append(metrics[m])
            n_win += 1
            if limit_windows is not None and n_win >= limit_windows:
                break
        print(f"  {path.name}: windows={n_win}")
    return values


def percentiles(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="SDP energy gate 캘리브레이션")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    ap.add_argument("--patterns", nargs="+", default=list(DEFAULT_PATTERNS))
    ap.add_argument("--metric", choices=list(ENERGY_METRICS), default="sdp_mean_abs")
    ap.add_argument("--multiplier", type=float, default=1.2,
                    help="recommended threshold = static p99 * multiplier")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--limit-windows-per-file", type=int, default=None)
    ap.add_argument("--fall-patterns", nargs="+", default=None,
                    help="fall energy p5 계산용 (예: *FALL*.csv). 미지정 시 생략.")
    ap.add_argument("--out", type=Path, default=None, help="결과 JSON 저장 경로(optional)")
    args = ap.parse_args()

    paths = iter_paths(args.root, args.patterns, args.limit_files)
    if not paths:
        raise SystemExit(f"No CSV matched {args.patterns} under {args.root}")

    print(f"=== static energy 수집 ({len(paths)} files) ===")
    static_vals = collect_energy(paths, args.limit_windows_per_file)
    static_stats = {m: percentiles(v) for m, v in static_vals.items()}

    sel = args.metric
    sel_stats = static_stats[sel]
    if sel_stats.get("n", 0) == 0:
        raise SystemExit(f"selected metric {sel!r} 에 수집된 윈도우 없음")

    recommended = sel_stats["p99"] * args.multiplier

    print(f"\n=== static metric 분포 (selected: {sel}) ===")
    for m in ENERGY_METRICS:
        s = static_stats[m]
        if s.get("n", 0) == 0:
            continue
        mark = " <== selected" if m == sel else ""
        print(
            f"  {m:<14} n={s['n']:<5} p50={s['p50']:.6f} p90={s['p90']:.6f} "
            f"p95={s['p95']:.6f} p99={s['p99']:.6f} max={s['max']:.6f}{mark}"
        )

    fall_stats = None
    fall_p5 = None
    if args.fall_patterns:
        fall_paths = iter_paths(args.root, args.fall_patterns, args.limit_files)
        if not fall_paths:
            print(f"\n[fall] {args.fall_patterns} 매칭 없음 — fall p5 생략")
        else:
            print(f"\n=== fall energy 수집 ({len(fall_paths)} files) ===")
            fall_vals = collect_energy(fall_paths, args.limit_windows_per_file)
            fall_stats = {m: percentiles(v) for m, v in fall_vals.items()}
            fall_p5 = fall_stats[sel].get("p5")

    print("\n=== 권장 threshold ===")
    print(f"  metric              : {sel}")
    print(f"  static p99          : {sel_stats['p99']:.6f}")
    print(f"  multiplier          : {args.multiplier}")
    print(f"  recommended thresh  : {recommended:.6f}  (= static p99 * multiplier)")
    if fall_p5 is not None:
        margin_ok = recommended < fall_p5
        print(f"  fall p5 ({sel})     : {fall_p5:.6f}")
        print(
            f"  threshold < fall p5 : {margin_ok}  "
            f"({'OK — fall 통과 여유 있음' if margin_ok else 'WARN — threshold 가 fall p5 이상! 낙상 차단 위험'})"
        )

    print("\n=== PowerShell 적용 예시 ===")
    print('  $env:SAFESIGNAL_ENERGY_GATE_ENABLED="1"')
    print(f'  $env:SAFESIGNAL_ENERGY_GATE_METRIC="{sel}"')
    print(f'  $env:SAFESIGNAL_ENERGY_GATE_THRESHOLD="{recommended:.6f}"')

    if args.out:
        payload = {
            "metric": sel,
            "multiplier": args.multiplier,
            "recommended_threshold": recommended,
            "static": static_stats,
            "fall": fall_stats,
            "fall_p5": fall_p5,
            "patterns": args.patterns,
            "fall_patterns": args.fall_patterns,
            "root": str(args.root),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[report] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
