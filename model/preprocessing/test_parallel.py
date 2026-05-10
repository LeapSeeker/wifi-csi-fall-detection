"""preprocess_directory_full 병렬 vs 단일 처리 검증.

실행:
    python -m model.preprocessing.test_parallel

작은 서브셋(Subject_1의 fall A2 = 20개)으로 병렬/단일 결과 일치성 + 속도를 비교.
Windows multiprocessing 호환을 위해 if __name__ == "__main__" 가드 사용.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from model.preprocessing import preprocess_directory_full


def main() -> int:
    root = (Path(__file__).resolve().parents[2]
            / "data" / "alsaify-raw" / "Environment 1" / "Subject_1")
    if not root.exists():
        print(f"FAIL: {root} 없음")
        return 1

    # A02만 (20개) → 빠른 검증
    pattern = "E1_S01_C01_A02_T*.csv"
    expected_n = 20

    print(f"테스트 디렉터리: {root}")
    print(f"패턴: {pattern}\n")

    # 1) 단일 프로세스
    print("[1/2] n_workers=1 (단일 프로세스)")
    t0 = time.time()
    serial = preprocess_directory_full(
        root, pattern=pattern, n_workers=1, show_progress=True
    )
    t_serial = time.time() - t0
    print(f"  → {len(serial)} 결과, {t_serial:.1f}s\n")

    # 2) 병렬
    n_workers = 4
    print(f"[2/2] n_workers={n_workers} (병렬)")
    t0 = time.time()
    parallel = preprocess_directory_full(
        root, pattern=pattern, n_workers=n_workers, show_progress=True
    )
    t_parallel = time.time() - t0
    print(f"  → {len(parallel)} 결과, {t_parallel:.1f}s\n")

    # 검증 1: 결과 개수
    assert len(serial) == expected_n, f"serial count {len(serial)} != {expected_n}"
    assert len(parallel) == expected_n, f"parallel count {len(parallel)} != {expected_n}"

    # 검증 2: shape 일관성
    for r in serial + parallel:
        assert r.inputs.shape[1:] == (1, 28, 20), f"shape {r.inputs.shape}"
        assert r.inputs.dtype == np.float32

    # 검증 3: 같은 (env,subj,activity,trial) 결과끼리 수치 일치
    serial_map = {(r.meta.environment, r.meta.subject, r.meta.activity,
                   r.meta.trial): r.inputs for r in serial}
    mismatches = 0
    for r in parallel:
        key = (r.meta.environment, r.meta.subject, r.meta.activity, r.meta.trial)
        diff = np.abs(serial_map[key] - r.inputs).max()
        if diff > 1e-5:
            mismatches += 1
            print(f"  MISMATCH {key}: max|Δ|={diff:.2e}")
    assert mismatches == 0, f"{mismatches} 개 결과 불일치"

    # 검증 4: 병렬이 단일보다 빠른지 (트라이얼이 작아 garante 못하지만 보통)
    speedup = t_serial / t_parallel
    print("=" * 60)
    print(f"✓ 결과 개수 일치: serial={len(serial)} parallel={len(parallel)}")
    print(f"✓ shape/dtype 일치: 모든 결과 (1, 28, 20) float32")
    print(f"✓ 수치 일치: max|Δ| < 1e-5 (모든 {expected_n}개)")
    print(f"  속도: serial {t_serial:.1f}s vs parallel({n_workers}w) {t_parallel:.1f}s "
          f"→ {speedup:.2f}× speedup")
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
