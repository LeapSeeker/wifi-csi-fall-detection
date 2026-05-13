"""CSI CSV 로딩.

지원 포맷:
1) Alsaify (Intel 5300 CSI Tool):
   - 13개 메타데이터 컬럼 (timestamp_low, bfee_count, Nrx, Ntx, rssi_*, noise, agc, perm_*, rate)
   - 90개 CSI 컬럼: csi_1_{1..3}_{1..30}  (3 안테나 x 30 서브캐리어)
   - 복소수 표기: "15+15i", "15+-7i" 등 (허수부 음수는 "+-" 형태)
   - 파일명 규칙: E{env}_S{subj}_C{class}_A{act}_T{trial}.csv

2) SafeSignal 자체수집 (collect/recorder.py 출력):
   - 107개 컬럼: timestamp_us, seq_rx{1,2}, amp_rx1_{0..51}, amp_rx2_{0..51}
   - 진폭만 저장(이미 sqrt(I^2+Q^2) 적용된 float32). 복소수 파싱 불필요.
   - 파일명 규칙: E{env}_S{subj}_A_{ACTIVITY}_T{trial}.csv  (예: E1_S01_A_STAND_T001)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

META_COLS = [
    "timestamp_low", "bfee_count", "Nrx", "Ntx",
    "rssi_a", "rssi_b", "rssi_c", "noise", "agc",
    "perm_1", "perm_2", "perm_3", "rate",
]
N_ANTENNAS = 3
N_SUBCARRIERS = 30
N_CSI = N_ANTENNAS * N_SUBCARRIERS  # 90

_FNAME_RE = re.compile(
    r"E(?P<env>\d+)_S(?P<subj>\d+)_C(?P<cls>\d+)_A(?P<act>\d+)_T(?P<trial>\d+)",
    re.IGNORECASE,
)
# "15+-7i" / "15+7i" / "-15+7i" 모두 처리
_COMPLEX_RE = re.compile(r"^(-?\d+)\+(-?\d+)i$")


@dataclass(frozen=True)
class AlsaifyMeta:
    environment: int
    subject: int
    class_id: int
    activity: int
    trial: int
    filename: str


def parse_alsaify_filename(path: str | Path) -> AlsaifyMeta:
    name = Path(path).stem
    m = _FNAME_RE.search(name)
    if not m:
        raise ValueError(f"Alsaify 파일명 규칙에 맞지 않음: {name}")
    return AlsaifyMeta(
        environment=int(m["env"]),
        subject=int(m["subj"]),
        class_id=int(m["cls"]),
        activity=int(m["act"]),
        trial=int(m["trial"]),
        filename=Path(path).name,
    )


def _parse_complex(token: str) -> complex:
    m = _COMPLEX_RE.match(token.strip())
    if not m:
        raise ValueError(f"복소수 파싱 실패: {token!r}")
    return complex(int(m.group(1)), int(m.group(2)))


def _csi_columns() -> list[str]:
    return [
        f"csi_1_{ant}_{sc}"
        for ant in range(1, N_ANTENNAS + 1)
        for sc in range(1, N_SUBCARRIERS + 1)
    ]


def load_csi_csv(path: str | Path) -> np.ndarray:
    """Alsaify CSV 한 파일을 복소수 ndarray로 로드.

    Returns
    -------
    csi : np.ndarray, shape (n_packets, 90), dtype=complex64
        축 0 = 패킷(시간), 축 1 = (안테나, 서브캐리어) flatten.
        순서: ant1[sc1..sc30], ant2[sc1..sc30], ant3[sc1..sc30]
    """
    cols = _csi_columns()
    df = pd.read_csv(path, usecols=cols)
    # 컬럼 순서 보장
    df = df[cols]

    n_packets = len(df)
    out = np.empty((n_packets, N_CSI), dtype=np.complex64)
    raw = df.to_numpy(dtype=object)

    for j in range(N_CSI):
        col = raw[:, j]
        for i in range(n_packets):
            out[i, j] = _parse_complex(col[i])
    return out


# ─── SafeSignal 자체수집 ───────────────────────────────────────────────────
# Alsaify 경로와 분리. 공통 abstraction 억지로 만들지 않는다.

SAFESIGNAL_N_SUBCARRIERS = 52  # LLTF, D-007

_SAFESIGNAL_FNAME_RE = re.compile(
    r"E(?P<env>\d+)_S(?P<subj>\d+)_A_(?P<act>[A-Z0-9_]+)_T(?P<trial>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SafeSignalMeta:
    environment: int
    subject: int
    activity: str       # "STAND", "WALK", ... 문자열 그대로 보존
    trial: int
    filename: str


@dataclass(frozen=True)
class SafeSignalRaw:
    amplitude: np.ndarray      # (n_packets, n_sc) float32 — rx별/concat에 따라
    timestamps_us: np.ndarray  # (n_packets,) int64
    rx: str                    # "rx1" | "rx2" | "both"
    meta: SafeSignalMeta


def parse_safesignal_filename(path: str | Path) -> SafeSignalMeta:
    name = Path(path).stem
    m = _SAFESIGNAL_FNAME_RE.search(name)
    if not m:
        raise ValueError(f"SafeSignal 파일명 규칙에 맞지 않음: {name}")
    return SafeSignalMeta(
        environment=int(m["env"]),
        subject=int(m["subj"]),
        activity=m["act"].upper(),
        trial=int(m["trial"]),
        filename=Path(path).name,
    )


def _safesignal_amp_columns(rx_idx: int) -> list[str]:
    return [f"amp_rx{rx_idx}_{i}" for i in range(SAFESIGNAL_N_SUBCARRIERS)]


def load_safesignal_csv(
    path: str | Path,
    rx: str = "both",
) -> SafeSignalRaw:
    """SafeSignal 자체수집 CSV 로드.

    Parameters
    ----------
    path : CSV 경로.
    rx : "rx1" | "rx2" | "both"
        - "rx1" → amplitude shape (n, 52)
        - "rx2" → amplitude shape (n, 52)
        - "both" → Rx1/Rx2 서브캐리어 concat (D-013), shape (n, 104)

    Returns
    -------
    SafeSignalRaw(amplitude, timestamps_us, rx, meta).

    필수 컬럼이 누락되면 ValueError. 파일명 규칙 위반은 parse_safesignal_filename
    가 ValueError를 발생.
    """
    if rx not in ("rx1", "rx2", "both"):
        raise ValueError(f"rx는 'rx1'|'rx2'|'both' 중 하나. got {rx!r}")

    meta = parse_safesignal_filename(path)

    if rx == "rx1":
        cols = ["timestamp_us"] + _safesignal_amp_columns(1)
    elif rx == "rx2":
        cols = ["timestamp_us"] + _safesignal_amp_columns(2)
    else:
        cols = (
            ["timestamp_us"]
            + _safesignal_amp_columns(1)
            + _safesignal_amp_columns(2)
        )

    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"SafeSignal CSV 필수 컬럼 누락: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''} (path={path})"
        )

    timestamps_us = df["timestamp_us"].to_numpy(dtype=np.int64)
    amp_cols = cols[1:]
    amplitude = df[amp_cols].to_numpy(dtype=np.float32)

    return SafeSignalRaw(
        amplitude=amplitude,
        timestamps_us=timestamps_us,
        rx=rx,
        meta=meta,
    )
