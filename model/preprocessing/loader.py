"""Alsaify CSI CSV 로딩.

Intel 5300 CSI Tool 포맷:
- 13개 메타데이터 컬럼 (timestamp_low, bfee_count, Nrx, Ntx, rssi_*, noise, agc, perm_*, rate)
- 90개 CSI 컬럼: csi_1_{1..3}_{1..30}  (3 안테나 x 30 서브캐리어)
- 복소수 표기: "15+15i", "15+-7i" 등 (허수부 음수는 "+-" 형태)

파일명 규칙: E{env}_S{subj}_C{class}_A{act}_T{trial}.csv
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
