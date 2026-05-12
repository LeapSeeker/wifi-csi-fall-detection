"""슬라이딩 윈도우 옵션 조합별 동작 검증.

실행:
    python debug/preprocessing/check_window.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from model.preprocessing.window import sliding_windows, WINDOW_SIZE


def check_window_cases() -> None:
    print("\n[1] 윈도우 케이스 표")
    print(f"  {'n_packets':>10}  {'stride':>7}  {'옵션':>30}  {'기대':>6}  {'실제':>6}  결과")

    rng = np.random.default_rng(0)

    cases = [
        (400, 300, dict(drop_last=True),                     1),
        (400, 300, dict(drop_last=False),                    2),
        (400, 300, dict(tail_window=True),                   2),
        (250, 300, dict(drop_last=True),                     0),
        (250, 300, dict(pad_short=True),                     1),
        (294, 300, dict(pad_short=True),                     1),
        (600, 100, dict(drop_last=True),                     4),
    ]

    all_pass = True
    for n_packets, stride, kwargs, expected in cases:
        amp = rng.uniform(0, 1, (n_packets, 52)).astype(np.float32)
        out = sliding_windows(amp, window_size=WINDOW_SIZE, stride=stride, **kwargs)
        ok = out.shape[0] == expected
        if not ok:
            all_pass = False
        opt_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"  {n_packets:>10}  {stride:>7}  {opt_str:>30}  {expected:>6}  {out.shape[0]:>6}  {'PASS' if ok else 'FAIL'}")

    print(f"\n  전체: {'ALL PASS' if all_pass else 'SOME FAIL'}")


def check_tail_window_content() -> None:
    print("\n[2] tail 윈도우 내용 검증 (amplitude[-300:])")
    n_packets = 400
    rng = np.random.default_rng(1)
    amp = rng.uniform(0, 1, (n_packets, 52)).astype(np.float32)
    out = sliding_windows(amp, window_size=WINDOW_SIZE, stride=300, tail_window=True)
    # 두 번째 윈도우가 amp[-300:]과 같아야 함
    expected_tail = amp[-300:]
    ok = np.allclose(out[1], expected_tail)
    print(f"  tail 윈도우 == amp[-300:]: {'PASS' if ok else 'FAIL'}")


def check_padding_position() -> None:
    print("\n[3] zero-padding 위치 검증 (뒷부분)")
    n_packets = 250
    rng = np.random.default_rng(2)
    amp = rng.uniform(1, 2, (n_packets, 52)).astype(np.float32)  # 0보다 큰 값
    out = sliding_windows(amp, window_size=WINDOW_SIZE, stride=300, pad_short=True)
    # 앞 250행은 원본, 뒤 50행은 0
    front_ok = np.allclose(out[0, :n_packets], amp)
    back_ok = np.allclose(out[0, n_packets:], 0.0)
    print(f"  앞 {n_packets}행 == 원본: {'PASS' if front_ok else 'FAIL'}")
    print(f"  뒤 {WINDOW_SIZE - n_packets}행 == 0: {'PASS' if back_ok else 'FAIL'}")


def main() -> None:
    print("=== 슬라이딩 윈도우 검증 ===")
    check_window_cases()
    check_tail_window_content()
    check_padding_position()
    print("\nDONE")


if __name__ == "__main__":
    main()
