"""전처리 파이프라인 단계별 스모크 테스트.

실행:
    python -m model.preprocessing.test_pipeline
또는:
    python model/preprocessing/test_pipeline.py [csv_path]

각 단계의 shape/dtype을 출력해 파이프라인이 의도대로 흐르는지 검증.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 직접 실행(`python model/preprocessing/test_pipeline.py`, IDE Run 버튼 등) 지원.
# `python -m model.preprocessing.test_pipeline` 으로 실행하면 __package__이 채워져
# 이 블록은 건너뜀.
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from model.preprocessing import (
    N_LAGS,
    SUB_STRIDE,
    SUB_W,
    W_T,
    WINDOW_SIZE,
    autocorrelation_matrix,
    downsample_alsaify,
    load_csi_csv,
    parse_alsaify_filename,
    rpca_sparse,
    sliding_windows,
    stacked_doppler_profile,
    to_amplitude,
    window_to_model_input,
)

DEFAULT_CSV = (
    Path(__file__).resolve().parents[2]
    / "data" / "alsaify-raw" / "Environment 1" / "Subject_1" / "E1_S01_C01_A01_T01.csv"
)
SR_RAW = 320   # Hz, Alsaify
SR_TARGET = 100  # Hz, ESP32 배포 환경


def _hr(title: str) -> None:
    print(f"\n[{title}]")


def main(csv_path: Path) -> int:
    if not csv_path.exists():
        print(f"FAIL: CSV not found: {csv_path}")
        return 1

    print(f"테스트 대상: {csv_path}")
    meta = parse_alsaify_filename(csv_path)
    print(f"메타: {meta}")

    # ── 1) CSV 로딩 ────────────────────────────────────────────────────────
    _hr("1) load_csi_csv")
    t0 = time.time()
    csi = load_csi_csv(csv_path)
    print(f"  shape={csi.shape}  dtype={csi.dtype}  [{time.time()-t0:.2f}s]")
    assert csi.ndim == 2 and csi.shape[1] == 90, f"expected (n,90), got {csi.shape}"
    assert np.iscomplexobj(csi), f"expected complex, got {csi.dtype}"
    n_raw = csi.shape[0]

    # ── 2) 진폭 ────────────────────────────────────────────────────────────
    _hr("2) to_amplitude")
    amp = to_amplitude(csi)
    print(f"  shape={amp.shape}  dtype={amp.dtype}  "
          f"min={amp.min():.3f} max={amp.max():.3f} mean={amp.mean():.3f}")
    assert amp.shape == csi.shape, "amplitude shape mismatch"
    assert amp.dtype == np.float32

    # ── 3) 다운샘플 320→100Hz ──────────────────────────────────────────────
    _hr("3) downsample_alsaify (320Hz → 100Hz, ratio 5/16)")
    ds = downsample_alsaify(amp)
    expected = int(round(n_raw * SR_TARGET / SR_RAW))
    print(f"  shape={ds.shape}  dtype={ds.dtype}")
    print(f"  raw n={n_raw}  →  downsampled n={ds.shape[0]}  (expected≈{expected})")
    # resample_poly는 ceil(n*up/down) 정도로 떨어짐. 허용 오차 ±2 패킷.
    assert abs(ds.shape[0] - expected) <= 2, "downsample length off by >2"
    assert ds.shape[1] == 90, "subcarrier dim changed"

    # ── 4) 슬라이딩 윈도우 ──────────────────────────────────────────────────
    _hr(f"4) sliding_windows (W={WINDOW_SIZE} packets, stride=W)")
    windows = sliding_windows(ds)
    print(f"  shape={windows.shape}  dtype={windows.dtype}  n_windows={windows.shape[0]}")
    if windows.shape[0] == 0:
        print("  WARN: 윈도우 0개 — 다운샘플 길이가 300 미만임")
        return 1
    assert windows.shape[1:] == (WINDOW_SIZE, 90)

    # ── 5) RPCA ────────────────────────────────────────────────────────────
    _hr("5) RPCA")
    win0 = windows[0]
    print(f"  input window:        shape={win0.shape}  axes=(time, subcarrier)")

    t0 = time.time()
    sparse = rpca_sparse(win0)
    print(f"  rpca_sparse output:  shape={sparse.shape}  dtype={sparse.dtype}  "
          f"axes=(time, subcarrier)  [{time.time()-t0:.2f}s]")
    print(f"    nnz_frac={(np.abs(sparse) > 1e-6).mean():.3f}  "
          f"min={sparse.min():.3f} max={sparse.max():.3f}")
    assert sparse.shape == (WINDOW_SIZE, 90), \
        f"RPCA shape mismatch: expected (300,90) got {sparse.shape}"

    # ── 6) ACF 차원 순서 검증 ──────────────────────────────────────────────
    _hr("6) ACF (sub-window 1개)")
    sub0 = sparse[:SUB_W]                                  # (30, 90)
    print(f"  sub-window slice:    shape={sub0.shape}  axes=(time, subcarrier)")
    acf_one = autocorrelation_matrix(sub0)                 # (90, 20)
    print(f"  ACF output:          shape={acf_one.shape}  axes=(subcarrier, lag)")
    print(f"    => ACF가 axis 순서를 (time, sc) → (sc, lag) 로 transpose함")
    print(f"    rho_0[:, 0] mean={float(acf_one[:, 0].mean()):.6f} (정확히 1.0이어야)")
    print(f"    rho_0[:, 0] min={float(acf_one[:, 0].min()):.6f} "
          f"max={float(acf_one[:, 0].max()):.6f}")
    assert acf_one.shape == (90, N_LAGS), \
        f"ACF shape mismatch: expected (90,{N_LAGS}) got {acf_one.shape}"
    assert np.allclose(acf_one[:, 0], 1.0, atol=1e-5), \
        "ACF lag-0이 1이 아님 → 축 0이 lag로 잘못 잡힘 가능성"

    # ── 7) SDP 입력 직전 / 집계 후 ─────────────────────────────────────────
    _hr("7) SDP (서브캐리어 축 평균 집계)")
    print(f"  SDP input:           shape={sparse.shape}  axes=(time, subcarrier)")
    print(f"  서브윈도우 분할:      W={SUB_W}, stride={SUB_STRIDE}, "
          f"n_sub=(300-30)/10+1={W_T}")

    sdp = stacked_doppler_profile(sparse)                  # (28, 20)
    print(f"  SDP output:          shape={sdp.shape}  axes=(sub_window, lag)")
    print(f"    => 각 sub-window: (sc=90, lag=20) → mean(axis=0) → (lag=20)")
    print(f"    값 범위: min={sdp.min():.3f} max={sdp.max():.3f} "
          f"mean={sdp.mean():.3f}")
    assert sdp.shape == (W_T, N_LAGS)

    # 첫 sub-window를 수동 재계산해 SDP[0]과 일치하는지 검증 (집계 축 정합)
    manual = autocorrelation_matrix(sparse[0:SUB_W]).mean(axis=0)  # (20,)
    print(f"  manual ACF(sub0).mean(axis=0) shape={manual.shape}")
    print(f"  || SDP[0] - manual || = {np.abs(sdp[0] - manual).max():.2e}")
    assert np.allclose(sdp[0], manual, atol=1e-5), \
        "SDP[0]과 mean(axis=0) 결과 불일치 → 집계 축이 잘못됨"
    print("  → SDP가 axis=0(서브캐리어)을 평균 집계함을 확인")

    model_input = window_to_model_input(win0)
    print(f"  window_to_model_input: shape={model_input.shape}  "
          f"dtype={model_input.dtype}")
    assert model_input.shape == (1, W_T, N_LAGS)
    assert model_input.dtype == np.float32

    # ── 8) 모델 forward (배치) ─────────────────────────────────────────────
    _hr("8) CNNGRUAttention forward — batch (4, 1, 28, 20)")
    try:
        import torch
        from model.pretrained.model import CLASSES, CNNGRUAttention, count_parameters
    except ImportError as e:
        print(f"  SKIP: {e}")
        print("\nPASS (전처리만) — 모델 모듈 로드 실패")
        return 0

    torch.manual_seed(0)
    model = CNNGRUAttention()
    model.eval()
    print(f"  model params: {count_parameters(model):,}")
    print(f"  classes:      {CLASSES}")

    x = torch.randn(4, 1, W_T, N_LAGS, dtype=torch.float32)
    print(f"  input:        shape={tuple(x.shape)}  dtype={x.dtype}")
    with torch.no_grad():
        logits, attn = model(x, return_attention=True)
        probs = torch.softmax(logits, dim=1)
    print(f"  logits:       shape={tuple(logits.shape)}  "
          f"(expected (4, {len(CLASSES)}))")
    print(f"  attn:         shape={tuple(attn.shape)}  "
          f"(expected (4, {W_T}))")
    attn_sums = attn.sum(dim=1).tolist()
    print(f"  attn softmax sums per sample: "
          f"{[f'{s:.4f}' for s in attn_sums]} (각 1.0)")
    print(f"  argmax labels: "
          f"{[CLASSES[i] for i in probs.argmax(dim=1).tolist()]}")
    assert logits.shape == (4, len(CLASSES))
    assert attn.shape == (4, W_T)
    assert all(abs(s - 1.0) < 1e-4 for s in attn_sums)

    # 실제 전처리 입력으로 한 번 더 (전처리 → 모델 호환성)
    real = window_to_model_input(win0)                     # (1, 28, 20)
    real_t = torch.from_numpy(real).unsqueeze(0)           # (1, 1, 28, 20)
    with torch.no_grad():
        real_logits = model(real_t)
    print(f"  real preprocessed input → logits shape={tuple(real_logits.shape)}  "
          f"argmax={CLASSES[int(real_logits.argmax(dim=1))]}")
    assert real_logits.shape == (1, len(CLASSES))

    print("\nPASS — 전처리 + 모델 forward 모두 통과")
    print(f"  최종 모델 입력 shape: (N, 1, {W_T}, {N_LAGS})")
    print(f"  최종 모델 출력 shape: (N, {len(CLASSES)})")
    return 0


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    sys.exit(main(arg))
