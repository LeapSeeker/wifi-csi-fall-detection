"""window_to_model_input() 단일 윈도우 latency 측정.

실행:
    python debug/preprocessing/benchmark_pipeline.py
    python debug/preprocessing/benchmark_pipeline.py --n 30 --n_sc 104
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model.preprocessing.pipeline import window_to_model_input


def benchmark(n: int, n_sc: int, max_iter: int) -> dict:
    rng = np.random.default_rng(42)
    window = rng.uniform(0, 1, (300, n_sc)).astype(np.float32)

    # 워밍업
    window_to_model_input(window, rpca_max_iter=max_iter)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        window_to_model_input(window, rpca_max_iter=max_iter)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        "mean": float(times.mean()),
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="파이프라인 latency 벤치마크")
    parser.add_argument("--n", type=int, default=10, help="측정 횟수")
    parser.add_argument("--n_sc", type=int, default=104, help="서브캐리어 수")
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()

    print("=== 파이프라인 Latency 벤치마크 ===")
    print(f"입력 shape: (300, {args.n_sc})  max_iter: {args.max_iter}  N={args.n}")
    print("워밍업 1회 실행 중...")

    result = benchmark(args.n, args.n_sc, args.max_iter)

    print(f"\n[결과]")
    print(f"  mean  = {result['mean']:.3f}s")
    print(f"  p50   = {result['p50']:.3f}s")
    print(f"  p95   = {result['p95']:.3f}s")

    print(f"\n[stride 기준 판정]")
    stride_limits = [(100, 1.0), (30, 0.3)]
    for stride, limit in stride_limits:
        ok = result["p95"] <= limit
        status = "PASS" if ok else "FAIL"
        print(f"  stride={stride:<4} 허용 한계 {limit:.2f}s (p95 기준): {status}")
        if not ok:
            print(f"         → max_iter 축소 또는 stride 증가 검토 필요")

    print(f"\n  ※ RTX4060에서 반드시 재측정 필요 (현재 결과는 CPU 기준)")
    print("\nDONE")


if __name__ == "__main__":
    main()
