"""Alsaify 전처리 파이프라인 오케스트레이터.

CSV → 진폭 → 100Hz 다운샘플 → 300패킷 윈도우 → RPCA → ACF → SDP → (1, 28, 20).
모델 입력 shape (N, 1, 28, 20)의 (1, 28, 20) 부분을 윈도우당 1개 생성.

병렬 모드 (preprocess_directory_full):
  RPCA가 파일당 ~2초로 가장 무거운 단계 → ProcessPoolExecutor로 파일 단위 병렬 처리.
  Windows에서는 호출 코드를 반드시 ``if __name__ == "__main__":`` 블록 안에서 실행.
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .acf import N_LAGS
from .amplitude import to_amplitude
from .downsample import downsample_alsaify
from .loader import (
    AlsaifyMeta,
    SafeSignalMeta,
    load_csi_csv,
    load_safesignal_csv,
    parse_alsaify_filename,
)
from .resample import ResampleResult, resample_to_100hz
from .rpca import DEFAULT_MAX_ITER, rpca_sparse
from .sdp import SUB_STRIDE, SUB_W, W_T, stacked_doppler_profile
from .window import WINDOW_SIZE, sliding_windows


@dataclass
class PreprocessResult:
    windows: np.ndarray  # (n_windows, 300, 90) float32
    meta: AlsaifyMeta


@dataclass
class ModelInputResult:
    inputs: np.ndarray   # (n_windows, 1, 28, 20) float32 — CNN 입력
    meta: AlsaifyMeta


# ─── 단계별 ──────────────────────────────────────────────────────────────────

def preprocess_file(
    path: str | Path,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    drop_last: bool = True,
    tail_window: bool = False,
    pad_short: bool = False,
) -> PreprocessResult:
    """CSV → 윈도우 텐서. RPCA 이전 단계까지.

    tail_window : True면 슬라이딩 후 잔여 패킷이 있을 때
    amplitude[-window_size:] 슬라이스 윈도우 1개를 추가한다.
    pad_short : True면 다운샘플 결과가 window_size 미만이어도 zero-padding으로
    윈도우 1개를 생성한다.
    """
    path = Path(path)
    meta = parse_alsaify_filename(path)

    csi = load_csi_csv(path)              # (m, 90) complex64
    amp = to_amplitude(csi)               # (m, 90) float32
    amp_100hz = downsample_alsaify(amp)   # (~m*5/16, 90) float32
    windows = sliding_windows(
        amp_100hz,
        window_size=window_size,
        stride=stride,
        drop_last=drop_last,
        tail_window=tail_window,
        pad_short=pad_short,
    )
    return PreprocessResult(windows=windows, meta=meta)


def window_to_model_input(
    window: np.ndarray,
    rpca_max_iter: int = DEFAULT_MAX_ITER,
    rpca_tol: float | None = None,
    sub_w: int = SUB_W,
    sub_stride: int = SUB_STRIDE,
    n_lags: int = N_LAGS,
) -> np.ndarray:
    """단일 윈도우 (300, n_sc) → 모델 입력 (1, 28, 20).

    1) RPCA → S(sparse) 성분
    2) 서브윈도우 ACF (lag=0..N_LAGS-1, 즉 lag0 포함) → 서브캐리어 평균 → SDP (28, 20)
    3) Global z-score 정규화 (윈도우 단위: (sdp - sdp.mean()) / (sdp.std() + 1e-6))
    4) 채널 축 추가 → (1, 28, 20)
    """
    if window.ndim != 2:
        raise ValueError(f"2D 입력 필요 (n_t, n_sc). got {window.shape}")
    sparse = rpca_sparse(window, max_iter=rpca_max_iter, tol=rpca_tol)
    sdp = stacked_doppler_profile(
        sparse, sub_w=sub_w, stride=sub_stride, n_lags=n_lags
    )
    sdp = (sdp - sdp.mean()) / (sdp.std() + 1e-6)
    return sdp[None, ...]  # (1, W_T, n_lags)


def windows_to_model_input(
    windows: np.ndarray,
    rpca_max_iter: int = DEFAULT_MAX_ITER,
    rpca_tol: float | None = None,
) -> np.ndarray:
    """배치 변환 (n_windows, 300, n_sc) → (n_windows, 1, 28, 20)."""
    if windows.ndim != 3:
        raise ValueError(f"3D 입력 필요 (n_windows, n_t, n_sc). got {windows.shape}")
    if windows.shape[0] == 0:
        return np.empty((0, 1, W_T, N_LAGS), dtype=np.float32)
    out = np.empty((windows.shape[0], 1, W_T, N_LAGS), dtype=np.float32)
    for i, w in enumerate(windows):
        out[i] = window_to_model_input(
            w, rpca_max_iter=rpca_max_iter, rpca_tol=rpca_tol
        )
    return out


# ─── 통합 ────────────────────────────────────────────────────────────────────

def preprocess_file_full(
    path: str | Path,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    drop_last: bool = True,
    tail_window: bool = False,
    pad_short: bool = False,
    rpca_max_iter: int = DEFAULT_MAX_ITER,
    rpca_tol: float | None = None,
) -> ModelInputResult:
    """단일 CSV → 모델 입력 텐서 (n_windows, 1, 28, 20).

    tail_window : True면 슬라이딩 후 잔여 패킷이 있을 때
    amplitude[-window_size:] 슬라이스 윈도우 1개를 추가한다.
    pad_short : True면 다운샘플 결과가 window_size 미만이어도 zero-padding으로
    윈도우 1개를 생성한다.
    """
    pre = preprocess_file(
        path,
        window_size=window_size,
        stride=stride,
        drop_last=drop_last,
        tail_window=tail_window,
        pad_short=pad_short,
    )
    inputs = windows_to_model_input(
        pre.windows, rpca_max_iter=rpca_max_iter, rpca_tol=rpca_tol
    )
    return ModelInputResult(inputs=inputs, meta=pre.meta)


def preprocess_directory(
    root: str | Path,
    pattern: str = "**/*.csv",
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    drop_last: bool = True,
    tail_window: bool = False,
    pad_short: bool = False,
) -> Iterable[PreprocessResult]:
    """디렉터리 내 CSV 순차 처리 (윈도우 단계까지). 제너레이터.

    대용량(피험자 20 × 활동 12 × 트라이얼 20 = 4,800파일)을 메모리에 한번에
    올리지 않도록 yield. RPCA까지 돌리려면 preprocess_directory_full.

    tail_window : True면 잔여 패킷이 있을 때 amplitude[-window_size:]
    슬라이스 윈도우 1개를 추가.
    pad_short : True면 다운샘플 결과가 window_size 미만이어도 zero-padding으로
    윈도우 1개를 생성한다.
    """
    root = Path(root)
    for csv_path in sorted(root.glob(pattern)):
        try:
            yield preprocess_file(
                csv_path,
                window_size=window_size,
                stride=stride,
                drop_last=drop_last,
                tail_window=tail_window,
                pad_short=pad_short,
            )
        except Exception as e:
            print(f"[skip] {csv_path}: {e}")


# ─── 병렬 풀 파이프라인 (RPCA 포함) ────────────────────────────────────────

def _worker_full(
    args: tuple[str, int, int | None, bool, bool, bool, int, float | None],
) -> tuple[str, ModelInputResult | None, str | None]:
    """ProcessPoolExecutor 워커. 모듈 최상위에 있어야 picklable.

    Returns
    -------
    (path, result | None, error_msg | None)
    """
    (
        path,
        window_size,
        stride,
        drop_last,
        tail_window,
        pad_short,
        rpca_max_iter,
        rpca_tol,
    ) = args
    try:
        res = preprocess_file_full(
            path,
            window_size=window_size,
            stride=stride,
            drop_last=drop_last,
            tail_window=tail_window,
            pad_short=pad_short,
            rpca_max_iter=rpca_max_iter,
            rpca_tol=rpca_tol,
        )
        return path, res, None
    except Exception as e:
        return path, None, repr(e)


def preprocess_files_full(
    paths: list[str | Path],
    n_workers: int | None = None,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    drop_last: bool = True,
    tail_window: bool = False,
    pad_short: bool = False,
    rpca_max_iter: int = DEFAULT_MAX_ITER,
    rpca_tol: float | None = None,
    show_progress: bool = True,
) -> list[ModelInputResult]:
    """주어진 CSV 목록을 풀 파이프라인(RPCA 포함)으로 병렬 처리. 핵심 함수.

    호출자가 미리 필터링한 명시적 파일 목록을 받음 (env/활동 필터링 등).
    디렉터리 글로브가 필요하면 ``preprocess_directory_full`` 사용.

    Parameters
    ----------
    paths : list[str | Path]
        처리할 CSV 파일 경로 목록.
    n_workers : int | None
        워커 프로세스 수. None → ``cpu_count()-1`` (최소 1).
        ≤1이면 단일 프로세스 (디버깅 / 작은 작업).
    window_size, stride, drop_last, tail_window, pad_short
        sliding_windows 파라미터. tail_window=True면 잔여 패킷이
        있을 때 amplitude[-window_size:] 슬라이스 윈도우 1개를 추가.
        pad_short=True면 다운샘플 결과가 window_size 미만이어도
        zero-padding으로 윈도우 1개를 생성한다.
    rpca_max_iter, rpca_tol
        RPCA 파라미터.
    show_progress : bool
        tqdm 진행 바.

    Returns
    -------
    list[ModelInputResult]
        실패 파일은 자동 스킵 (tqdm.write로 메시지 출력).
        병렬 모드에서는 결과 순서가 완료 순 (입력 paths 순서와 다를 수 있음).
        순서 필요 시 호출자가 ``meta`` 필드로 정렬.

    주의 (Windows / macOS spawn)
    ---------------------------
    호출 코드는 반드시 ``if __name__ == "__main__":`` 블록 안에서 실행할 것.
    그렇지 않으면 자식 프로세스가 부모 모듈을 다시 import 하면서 무한 spawn.
    """
    if not paths:
        return []

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    args_list = [
        (
            str(p),
            window_size,
            stride,
            drop_last,
            tail_window,
            pad_short,
            rpca_max_iter,
            rpca_tol,
        )
        for p in paths
    ]
    results: list[ModelInputResult] = []

    if n_workers <= 1:
        # 단일 프로세스 — fork/spawn 오버헤드 없음, 디버깅 용이
        iterable = tqdm(
            args_list,
            total=len(args_list),
            desc="preprocess (1 worker)",
            disable=not show_progress,
        )
        for args in iterable:
            path, res, err = _worker_full(args)
            if err is not None:
                tqdm.write(f"[skip] {path}: {err}")
                continue
            results.append(res)
        return results

    # 병렬
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_worker_full, a) for a in args_list]
        iterable = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"preprocess ({n_workers} workers)",
            disable=not show_progress,
        )
        for future in iterable:
            path, res, err = future.result()
            if err is not None:
                tqdm.write(f"[skip] {path}: {err}")
                continue
            results.append(res)
    return results


def preprocess_directory_full(
    root: str | Path,
    pattern: str = "**/*.csv",
    n_workers: int | None = None,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    drop_last: bool = True,
    tail_window: bool = False,
    pad_short: bool = False,
    rpca_max_iter: int = DEFAULT_MAX_ITER,
    rpca_tol: float | None = None,
    show_progress: bool = True,
) -> list[ModelInputResult]:
    """``root``를 ``pattern``으로 글로브 후 ``preprocess_files_full`` 호출.

    필터링이 더 필요하면 호출자가 직접 글로브 후 ``preprocess_files_full``.
    tail_window : True면 잔여 패킷이 있을 때 amplitude[-window_size:]
    슬라이스 윈도우 1개를 추가.
    pad_short : True면 다운샘플 결과가 window_size 미만이어도 zero-padding으로
    윈도우 1개를 생성한다.
    """
    paths = sorted(Path(root).glob(pattern))
    return preprocess_files_full(
        paths,
        n_workers=n_workers,
        window_size=window_size,
        stride=stride,
        drop_last=drop_last,
        tail_window=tail_window,
        pad_short=pad_short,
        rpca_max_iter=rpca_max_iter,
        rpca_tol=rpca_tol,
        show_progress=show_progress,
    )


# ─── SafeSignal 자체수집 전용 (D-018) ─────────────────────────────────────
# Alsaify 경로(downsample_alsaify)와 분리. 공통 후단(sliding_windows,
# windows_to_model_input)만 재사용한다.

@dataclass
class SafeSignalPreprocessResult:
    windows: np.ndarray              # (n_windows, window_size, n_sc) float32
    meta: SafeSignalMeta
    resample: ResampleResult         # 품질 metadata (gap_count, original_rate_hz 등)


@dataclass
class SafeSignalModelInputResult:
    inputs: np.ndarray               # (n_windows, 1, 28, 20) float32 — CNN 입력
    meta: SafeSignalMeta
    resample: ResampleResult


def preprocess_safesignal_file(
    path: str | Path,
    rx: str = "both",
    target_hz: float = 100.0,
    max_gap_ms: float = 100.0,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    drop_last: bool = True,
    tail_window: bool = False,
    pad_short: bool = False,
) -> SafeSignalPreprocessResult:
    """SafeSignal CSV → 100Hz 리샘플 → 윈도우.

    흐름: load_safesignal_csv → resample_to_100hz → sliding_windows.
    Alsaify의 downsample_alsaify(320→100Hz)와는 다른 경로(timestamp 기반 보간).

    rx="both" 기본 — Rx1/Rx2 concat (D-013), 윈도우 shape (?, 300, 104).
    5초 수집 데이터에서는 호출자가 tail_window=True를 줘 잔여 패킷 보존 가능.
    """
    raw = load_safesignal_csv(path, rx=rx)
    res = resample_to_100hz(
        raw.amplitude,
        raw.timestamps_us,
        target_hz=target_hz,
        max_gap_ms=max_gap_ms,
    )
    windows = sliding_windows(
        res.amplitude,
        window_size=window_size,
        stride=stride,
        drop_last=drop_last,
        tail_window=tail_window,
        pad_short=pad_short,
    )
    return SafeSignalPreprocessResult(
        windows=windows,
        meta=raw.meta,
        resample=res,
    )


def preprocess_safesignal_file_full(
    path: str | Path,
    rx: str = "both",
    target_hz: float = 100.0,
    max_gap_ms: float = 100.0,
    window_size: int = WINDOW_SIZE,
    stride: int | None = None,
    drop_last: bool = True,
    tail_window: bool = False,
    pad_short: bool = False,
    rpca_max_iter: int = DEFAULT_MAX_ITER,
    rpca_tol: float | None = None,
) -> SafeSignalModelInputResult:
    """SafeSignal CSV → 모델 입력 텐서 (n_windows, 1, 28, 20).

    preprocess_safesignal_file 결과에 windows_to_model_input(RPCA→ACF→SDP) 적용.
    """
    pre = preprocess_safesignal_file(
        path,
        rx=rx,
        target_hz=target_hz,
        max_gap_ms=max_gap_ms,
        window_size=window_size,
        stride=stride,
        drop_last=drop_last,
        tail_window=tail_window,
        pad_short=pad_short,
    )
    inputs = windows_to_model_input(
        pre.windows, rpca_max_iter=rpca_max_iter, rpca_tol=rpca_tol
    )
    return SafeSignalModelInputResult(
        inputs=inputs,
        meta=pre.meta,
        resample=pre.resample,
    )
