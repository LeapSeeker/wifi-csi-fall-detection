"""CSV → (1, 28, 20) 전체 파이프라인 단계별 shape/범위 추적.

실행:
    python debug/preprocessing/check_pipeline_e2e.py               # 합성 데이터
    python debug/preprocessing/check_pipeline_e2e.py --csv path/   # 실제 CSV
    python debug/preprocessing/check_pipeline_e2e.py --mode esp32  # (300, 104) 케이스
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model.preprocessing.rpca import rpca_sparse
from model.preprocessing.sdp import stacked_doppler_profile
from model.preprocessing.pipeline import window_to_model_input


def _check_step(step: int, name: str, arr: np.ndarray) -> bool:
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    nan_inf_str = f"NaN={has_nan} Inf={has_inf}"
    print(f"  [{step}] {name:<8} shape={str(arr.shape):<16} dtype={arr.dtype}  "
          f"range=[{arr.min():.4f}, {arr.max():.4f}]  {nan_inf_str}")
    return not (has_nan or has_inf)


def run_pipeline(window: np.ndarray, label: str) -> bool:
    print(f"\n--- {label} ---")
    ok = True

    ok &= _check_step(0, "입력", window)

    sparse = rpca_sparse(window, max_iter=50)
    ok &= _check_step(1, "RPCA", sparse)

    sdp = stacked_doppler_profile(sparse)
    ok &= _check_step(2, "SDP", sdp)

    ch = sdp[None, ...]
    ok &= _check_step(3, "채널", ch)

    final_ok = ch.shape == (1, 28, 20)
    print(f"  최종 모델 입력 shape: {ch.shape}  {'PASS' if final_ok else 'FAIL'}")

    # window_to_model_input 통합 함수 비교
    result2 = window_to_model_input(window, rpca_max_iter=50)
    unified_ok = np.allclose(ch, result2, atol=1e-5)
    print(f"  window_to_model_input() 일치: {'PASS' if unified_ok else 'FAIL'}")

    return ok and final_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="파이프라인 E2E 검증")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--mode", choices=["alsaify", "esp32"], default="alsaify")
    args = parser.parse_args()

    print("=== 파이프라인 E2E 검증 ===")

    rng = np.random.default_rng(42)

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        amp_cols = [c for c in df.columns if c.startswith("amp_rx")]
        window = df[amp_cols].values[:300].astype(np.float32)
        if window.shape[0] < 300:
            pad = np.zeros((300 - window.shape[0], window.shape[1]), dtype=np.float32)
            window = np.vstack([window, pad])
        run_pipeline(window, f"실제 CSV ({window.shape})")
    else:
        # Alsaify (300, 90)
        window_alsaify = rng.uniform(0, 1, (300, 90)).astype(np.float32)
        run_pipeline(window_alsaify, "합성 데이터 Alsaify (300, 90)")

        # ESP32 concat (300, 104)
        window_esp32 = rng.uniform(0, 1, (300, 104)).astype(np.float32)
        run_pipeline(window_esp32, "합성 데이터 ESP32 concat (300, 104)")

    print("\nDONE")


if __name__ == "__main__":
    main()
