"""clean400 더미 생성 공용 라이브러리 (read-only 입력).

- clean400 = Gate1 좌표계 concat(raw550[50:150]+[200:400]+[450:550]) (400,104).
- clean400 좌표 onset detector (gate2 동일 에너지: rpca_sparse→mean|·|→5smooth, clean baseline/search).
- time_warp/scale/noise/subcarrier 증강 (clean400 위에서), 난수 파라미터 전부 반환.
- onset 해석적 추적(expected) 계산.

동결파일·manifest·원본 무수정. 계산은 gate2 함수 재사용.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy import interpolate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "debug/modeling"))

from model.preprocessing.loader import load_safesignal_csv      # noqa: E402
from model.preprocessing.resample import resample_to_100hz      # noqa: E402
from model.preprocessing.rpca import rpca_sparse, DEFAULT_MAX_ITER  # noqa: E402

MIN_FRAMES = 550
CLEAN_LEN = 400
SMOOTH = 5
K_MAD = 3.0
SUSTAIN = 5
BASE_CLEAN = (0, 100)     # clean baseline = raw[50:150]
SEARCH_CLEAN = (100, 300)  # 유효 onset 탐색(non-WALK onset 100~250)
# clean400 splice 경계 (raw[50:150]|[200:400]|[450:550] 접합 → clean frame 100, 300)
SPLICE_BOUNDARIES = (100, 300)

BODY_PARAMS = {"small": (0.75, 0.90), "medium": (0.95, 1.05), "large": (1.10, 1.30)}
SPEED_PARAMS = {"fast": (0.75, 0.90), "normal": (0.95, 1.05), "slow": (1.10, 1.30)}
NOISE_SNR_DB = (25.0, 35.0)  # 보수적(팀원 18~28보다 높게 → baseline_noise flag 억제)


def build_clean400(csv_path):
    """원본 CSV → clean400 (400,104) 또는 None(resampled<550)."""
    raw = load_safesignal_csv(csv_path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    if int(res.resampled_count) < MIN_FRAMES:
        return None
    a = res.amplitude[:MIN_FRAMES]
    return np.concatenate([a[50:150], a[200:400], a[450:550]], axis=0).astype(np.float64)


def _smooth(e):
    h = SMOOTH // 2
    return np.array([e[max(0, i - h):min(len(e), i + h + 1)].mean() for i in range(len(e))])


def energy_clean(clean400):
    e = np.abs(rpca_sparse(clean400, max_iter=DEFAULT_MAX_ITER, tol=None)).mean(axis=1)
    return _smooth(e)


def detect_onset_clean(clean400, es=None):
    """clean400 좌표 onset 재검출. 반환 dict(rise, pob, baseline_noise_ratio, rise_not_found)."""
    if es is None:
        es = energy_clean(clean400)
    b0, b1 = BASE_CLEAN
    base = es[b0:b1]
    bm = float(np.median(base))
    bmad = float(np.median(np.abs(base - bm))) or 1e-9
    thr = bm + K_MAD * bmad
    s0, s1 = SEARCH_CLEAN
    s1 = min(s1, len(es))
    rise = None
    for i in range(s0, s1):
        j = i + SUSTAIN
        if j <= len(es) and np.all(es[i:j] > thr):
            rise = i
            break
    pob = float(es[s0:s1].max() / (bm + 1e-9)) if s1 > s0 else float("nan")
    noise = float(np.percentile(base, 90) / (bm + 1e-9))
    return {"rise": rise, "pob": pob, "baseline_noise_ratio": noise,
            "thr": thr, "bm": bm, "rise_not_found": rise is None}


# ── 증강 ──────────────────────────────────────────────────────────────────────
def _amplitude_scale(x, rng, lo, hi, scale=None):
    if scale is None:
        scale = float(rng.uniform(lo, hi))
    return x * scale, scale


def _time_warp(x, rng, lo, hi, factor=None):
    """uniform time-warp on clean400. fast(f<1) tail-pad / slow(f>1) center-crop(peak 보존).
    반환 (warped(L,104), factor, crop_offset)."""
    if factor is None:
        factor = float(rng.uniform(lo, hi))
    L = len(x)
    oi = np.linspace(0, 1, L)
    ni = np.linspace(0, 1, max(2, int(L * factor)))
    w = np.zeros((len(ni), x.shape[1]))
    for c in range(x.shape[1]):
        w[:, c] = interpolate.interp1d(oi, x[:, c])(ni)
    if len(w) >= L:
        trim = len(w) - L
        start = trim // 2                      # center-crop (peak 보존, 결정론적)
        return w[start:start + L], factor, start
    pad = np.tile(w[-1:], (L - len(w), 1))     # fast: tail = 마지막 프레임
    return np.vstack([w, pad]), factor, 0


def _subcarrier(x, rng, lo=0.95, hi=1.05):
    return x * rng.uniform(lo, hi, size=x.shape[1])


def _add_noise(x, rng, snr_lo, snr_hi):
    snr = float(rng.uniform(snr_lo, snr_hi))
    sp = np.mean(x ** 2)
    npow = sp / (10 ** (snr / 10))
    return x + rng.normal(0, np.sqrt(npow), x.shape), snr


def expected_onset(onset_clean, factor, crop_offset, L=CLEAN_LEN):
    """uniform warp + crop 의 결정론적 onset 매핑. None=범위밖(crop_out_of_bounds)."""
    new_len = max(2, int(L * factor))
    warped = round(onset_clean * (new_len - 1) / (L - 1))
    dummy = warped - crop_offset
    if dummy < 0 or dummy >= L:
        return None
    return int(dummy)


def augment_clean400(clean400, profile, rng):
    """profile={body_size,fall_speed} → (aug400, params). rx1/rx2 분리 적용(같은 scale/factor 공유)."""
    rx1 = clean400[:, :52].copy()
    rx2 = clean400[:, 52:].copy()
    blo, bhi = BODY_PARAMS[profile["body_size"]]
    slo, shi = SPEED_PARAMS[profile["fall_speed"]]
    rx1, scale = _amplitude_scale(rx1, rng, blo, bhi)
    rx2, _ = _amplitude_scale(rx2, rng, blo, bhi, scale=scale)
    rx1, factor, off = _time_warp(rx1, rng, slo, shi)
    rx2, _, _ = _time_warp(rx2, rng, slo, shi, factor=factor)
    rx1 = _subcarrier(rx1, rng)
    rx2 = _subcarrier(rx2, rng)
    rx1, snr = _add_noise(rx1, rng, *NOISE_SNR_DB)
    rx2, _ = _add_noise(rx2, rng, *NOISE_SNR_DB)
    aug = np.concatenate([rx1, rx2], axis=1)
    return aug, {"scale_factor": round(scale, 5), "warp_factor": round(factor, 5),
                 "crop_offset": int(off), "noise_snr": round(snr, 2)}
