"""SDP 텐서 ((1, 28, 20)) 시계열 증강.

4가지 기법 (model/CLAUDE.md 증강 기법: Jittering / Scaling / Time Warping / Noise+Scale):

| 기법         | 동작                                                          |
|-------------|---------------------------------------------------------------|
| jittering    | 가산성 가우시안 노이즈                                          |
| scaling      | 텐서 전체에 단일 무작위 스칼라 곱                              |
| time_warping | 시간 축(axis=1)을 cubic-spline 기반 무작위 곡선으로 왜곡       |
| noise_scale  | jittering + scaling 결합                                       |

모두 입력 shape/dtype을 보존. 자체 수집 240세션 × 5배 = 1,200샘플 목표용.

참고: Um et al. (2017), "Data Augmentation of Wearable Sensor Data
for Parkinson's Disease Monitoring using Convolutional Neural Networks."
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


# ── 개별 기법 ──────────────────────────────────────────────────────────────

def jittering(
    x: np.ndarray,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """가산성 가우시안 노이즈 N(0, σ²)."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=x.shape).astype(x.dtype, copy=False)
    return x + noise


def scaling(
    x: np.ndarray,
    scale_range: tuple[float, float] = (0.8, 1.2),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """텐서 전체에 단일 무작위 스칼라 곱 (uniform[scale_range])."""
    if rng is None:
        rng = np.random.default_rng()
    scale = float(rng.uniform(*scale_range))
    return (x * scale).astype(x.dtype, copy=False)


def time_warping(
    x: np.ndarray,
    sigma: float = 0.2,
    knot: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """시간 축(axis=1)을 cubic-spline 곡선으로 왜곡.

    knot+2개 제어점에서 N(1, σ²) 무작위 magnitude → 누적합으로 단조 증가
    매핑 생성 → 각 lag을 새 시간 격자에 ``np.interp``로 보간.
    np.interp는 외삽하지 않아 결과는 [x.min(), x.max()] 범위 안에 머무름.

    distortion이 음수가 되면 누적합 단조성이 깨지므로 1e-3으로 clamp.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_t = x.shape[1]

    knot_xs = np.linspace(0, n_t - 1, knot + 2)
    knot_ys = rng.normal(loc=1.0, scale=sigma, size=knot + 2)
    cs = CubicSpline(knot_xs, knot_ys)
    distortion = np.maximum(cs(np.arange(n_t)), 1e-3)

    tt_cum = np.cumsum(distortion)
    tt_cum = (tt_cum - tt_cum[0]) / (tt_cum[-1] - tt_cum[0]) * (n_t - 1)

    time_steps = np.arange(n_t)
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        for l in range(x.shape[2]):
            out[c, :, l] = np.interp(time_steps, tt_cum, x[c, :, l])
    return out


def noise_scale(
    x: np.ndarray,
    sigma: float = 0.05,
    scale_range: tuple[float, float] = (0.8, 1.2),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """jittering + scaling 결합 (noise → scale 순)."""
    if rng is None:
        rng = np.random.default_rng()
    return scaling(
        jittering(x, sigma=sigma, rng=rng),
        scale_range=scale_range,
        rng=rng,
    )


# ── 통합 ───────────────────────────────────────────────────────────────────

def augment_all(
    x: np.ndarray,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """원본 + 4기법 = 길이 5 리스트 (5배 증강).

    인덱스 0: 원본 (참조, copy 아님 — 수정 시 주의).
    인덱스 1~4: jittering, scaling, time_warping, noise_scale.

    개별 기법 파라미터 조정이 필요하면 각 함수를 직접 호출.
    """
    if rng is None:
        rng = np.random.default_rng()
    return [
        x,
        jittering(x, rng=rng),
        scaling(x, rng=rng),
        time_warping(x, rng=rng),
        noise_scale(x, rng=rng),
    ]


# ── 단위 테스트 ────────────────────────────────────────────────────────────

def _self_test() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(-1.0, 1.0, size=(1, 28, 20)).astype(np.float32)
    print(f"테스트 입력: shape={x.shape} dtype={x.dtype} "
          f"range=[{x.min():.3f}, {x.max():.3f}]")

    # 1) shape/dtype 보존
    for fn in (jittering, scaling, time_warping, noise_scale):
        y = fn(x, rng=np.random.default_rng(0))
        assert y.shape == (1, 28, 20), f"{fn.__name__} shape={y.shape}"
        assert y.dtype == np.float32, f"{fn.__name__} dtype={y.dtype}"
    print("  [1] shape/dtype 보존 ✓")

    # 2) jittering: 차이의 std가 sigma 근처
    y_j = jittering(x, sigma=0.05, rng=np.random.default_rng(0))
    diff_std = float((y_j - x).std())
    assert 0.03 < diff_std < 0.07, f"jitter std {diff_std:.4f} 가 ~0.05 이탈"
    print(f"  [2] jittering: 노이즈 std={diff_std:.4f} (≈sigma=0.05) ✓")

    # 3) scaling: 모든 비율이 동일한 단일 스칼라
    y_s = scaling(x, scale_range=(0.8, 1.2), rng=np.random.default_rng(0))
    nz = np.abs(x) > 1e-3
    ratios = y_s[nz] / x[nz]
    assert 0.8 <= ratios.mean() <= 1.2, f"scale={ratios.mean():.4f} 범위 이탈"
    assert ratios.std() < 1e-5, f"단일 scale 아님 std={ratios.std():.2e}"
    print(f"  [3] scaling: ratio={ratios.mean():.4f} (∈[0.8, 1.2]), "
          f"단일 스칼라 std={ratios.std():.2e} ✓")

    # 4) time_warping: np.interp이라 [min, max] 범위 안에 bound
    y_w = time_warping(x, rng=np.random.default_rng(0))
    assert y_w.min() >= x.min() - 1e-5, \
        f"warp min {y_w.min():.4f} < orig min {x.min():.4f}"
    assert y_w.max() <= x.max() + 1e-5, \
        f"warp max {y_w.max():.4f} > orig max {x.max():.4f}"
    # 실제 변형이 일어났는지 (원본과 다름)
    assert not np.allclose(y_w, x), "time_warping이 원본과 동일"
    print(f"  [4] time_warping: 출력 범위 [{y_w.min():.3f}, {y_w.max():.3f}] "
          f"⊂ 원본 [{x.min():.3f}, {x.max():.3f}] ✓")

    # 5) noise_scale: 변경 발생
    y_ns = noise_scale(x, rng=np.random.default_rng(0))
    diff_ns = float(np.abs(y_ns - x).mean())
    assert diff_ns > 0.0, "noise_scale 효과 없음"
    print(f"  [5] noise_scale: 평균 |Δ|={diff_ns:.4f} ✓")

    # 6) augment_all
    augs = augment_all(x, rng=np.random.default_rng(0))
    assert len(augs) == 5, f"augment_all 길이 {len(augs)} (기대 5)"
    assert augs[0] is x, "augment_all[0]이 원본 참조 아님"
    for i, a in enumerate(augs):
        assert a.shape == (1, 28, 20)
        assert a.dtype == np.float32
    # 1~4번은 모두 원본과 달라야 함
    for i in range(1, 5):
        assert not np.array_equal(augs[i], x), f"augment_all[{i}]가 원본과 동일"
    print(f"  [6] augment_all: 길이 {len(augs)}, [0]=원본, [1-4]=4 기법 모두 변형 ✓")

    # 7) 결정론 (같은 seed면 같은 결과)
    a1 = jittering(x, rng=np.random.default_rng(123))
    a2 = jittering(x, rng=np.random.default_rng(123))
    assert np.array_equal(a1, a2), "동일 seed에서 결과가 다름"
    print("  [7] 결정론: 동일 seed → 동일 결과 ✓")

    print("\nPASS — 7개 검증 모두 통과")


if __name__ == "__main__":
    _self_test()
