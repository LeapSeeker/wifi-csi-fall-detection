"""collect/ 모듈 self-check (단위 테스트 대용).

검증 항목:
1. UDP 패킷 파싱
2. Rx1/Rx2 timestamp 기반 페어링
3. 파일명 trial 번호 자동 증가
4. count_sessions
5. calculate_loss_rate
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import socket as _socket

from collect.recorder import SessionRecorder
from collect.udp import (
    AMPLITUDE_COUNT,
    DEVICE_ID_RX1,
    DEVICE_ID_RX2,
    HEADER_FORMAT,
    PACKET_MAGIC,
    PACKET_SIZE,
    CsiPacket,
    PairingBuffer,
    parse_packet,
    start_udp_receiver,
)


def make_packet_bytes(device_id: int, rssi: int, seq: int, ts_us: int, amp_seed: float) -> bytes:
    header = struct.pack(HEADER_FORMAT, PACKET_MAGIC, device_id, rssi, 0, seq, ts_us)
    amps = [amp_seed + i * 0.1 for i in range(AMPLITUDE_COUNT)]
    body = struct.pack(f"<{AMPLITUDE_COUNT}f", *amps)
    return header + body


def test_parse_packet() -> None:
    data = make_packet_bytes(DEVICE_ID_RX1, -42, 100, 1_000_000, 1.0)
    assert len(data) == PACKET_SIZE, f"size {len(data)}"
    pkt = parse_packet(data)
    assert pkt is not None
    assert pkt.device_id == DEVICE_ID_RX1
    assert pkt.rssi == -42
    assert pkt.seq == 100
    assert pkt.timestamp_us == 1_000_000
    assert len(pkt.amplitudes) == AMPLITUDE_COUNT
    assert abs(pkt.amplitudes[0] - 1.0) < 1e-5

    # bad magic
    bad = bytearray(data)
    bad[0] = 0x00
    assert parse_packet(bytes(bad)) is None

    # bad size
    assert parse_packet(data[:-1]) is None

    # bad device_id
    bad2 = bytearray(data)
    bad2[1] = 0xFF
    assert parse_packet(bytes(bad2)) is None
    print("[OK] parse_packet")


def test_pairing() -> None:
    paired: list[tuple[CsiPacket, CsiPacket]] = []
    pb = PairingBuffer(on_paired=lambda r1, r2: paired.append((r1, r2)))

    # rx1 먼저 들어옴, rx2가 30ms 뒤 → pair
    p1 = CsiPacket(DEVICE_ID_RX1, -40, 1, 1_000_000, [0.0] * AMPLITUDE_COUNT)
    p2 = CsiPacket(DEVICE_ID_RX2, -45, 1, 1_030_000, [0.0] * AMPLITUDE_COUNT)
    pb.add(p1)
    assert paired == []
    pb.add(p2)
    assert len(paired) == 1
    assert paired[0][0].device_id == DEVICE_ID_RX1
    assert paired[0][1].device_id == DEVICE_ID_RX2

    # 60ms 차이 → pair 실패 (50ms tolerance 초과)
    paired.clear()
    p3 = CsiPacket(DEVICE_ID_RX1, -40, 2, 2_000_000, [0.0] * AMPLITUDE_COUNT)
    p4 = CsiPacket(DEVICE_ID_RX2, -45, 2, 2_060_000, [0.0] * AMPLITUDE_COUNT)
    pb.add(p3)
    pb.add(p4)
    assert paired == [], "60ms 차이는 pair되면 안 됨"

    # cleanup으로 만료 패킷 제거
    pb.cleanup()  # latest_ts_us = 2_060_000, p3(2_000_000) diff=60_000us — under 200ms, 유지
    p5 = CsiPacket(DEVICE_ID_RX2, -40, 99, 2_500_000, [0.0] * AMPLITUDE_COUNT)
    pb.add(p5)  # latest_ts = 2_500_000, p3,p4 모두 expire
    pb.cleanup()
    print("[OK] pairing")


def test_loss_rate() -> None:
    # seq 0..9 모두 존재 → 0%
    buf = [{"seq_rx1": i, "seq_rx2": i} for i in range(10)]
    assert SessionRecorder.calculate_loss_rate(buf) == 0.0

    # rx1: 0..9 중 5 누락, rx2: 모두 → max=10%
    buf2 = [{"seq_rx1": i, "seq_rx2": i} for i in range(10) if i != 5]
    loss = SessionRecorder.calculate_loss_rate(buf2)
    # expected = 10-1+1 - 9 = 1 → 1/10 = 0.1
    assert abs(loss - 0.1) < 1e-9, f"loss={loss}"

    # 빈 버퍼 → 0
    assert SessionRecorder.calculate_loss_rate([]) == 0.0
    # 1개 → 0
    assert SessionRecorder.calculate_loss_rate([{"seq_rx1": 5, "seq_rx2": 5}]) == 0.0
    print("[OK] calculate_loss_rate")


def test_trial_and_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        rec = SessionRecorder(raw_dir=raw)

        # count: 0
        assert rec.count_sessions("WALK", env=1, subject=1) == 0

        # 가짜 세션 1건 저장 (rssi 컬럼 없음 — 110 컬럼, 신규 품질 컬럼은 누락 시 빈값)
        buf1 = [
            {col: 0 for col in ["timestamp_us", "seq_rx1", "seq_rx2"]}
        ]
        for i in range(AMPLITUDE_COUNT):
            buf1[0][f"amp_rx1_{i}"] = float(i)
            buf1[0][f"amp_rx2_{i}"] = float(i + 100)

        path1 = rec.save_session(buf1, "WALK", env=1, subject=1)
        assert path1.name == "E1_S01_A_WALK_T001.csv", path1.name
        assert rec.count_sessions("WALK", 1, 1) == 1

        path2 = rec.save_session(buf1, "WALK", env=1, subject=1)
        assert path2.name == "E1_S01_A_WALK_T002.csv", path2.name
        assert rec.count_sessions("WALK", 1, 1) == 2

        # 다른 env/subject는 영향 없음
        assert rec.count_sessions("WALK", 2, 1) == 0
        assert rec.count_sessions("WALK", 1, 2) == 0

        # CSV 컬럼 순서 / 개수 / rssi 부재 검증
        import pandas as pd

        df = pd.read_csv(path1)
        from collect.recorder import CSV_COLUMNS

        assert list(df.columns) == CSV_COLUMNS, "CSV column order mismatch"
        assert len(CSV_COLUMNS) == 110, f"expected 110 columns, got {len(CSV_COLUMNS)}"
        assert "rssi_rx1" not in df.columns and "rssi_rx2" not in df.columns

        empty_path = rec.save_session([], "WALK", env=1, subject=1)
        assert empty_path is None
        assert rec.count_sessions("WALK", 1, 1) == 2
    print("[OK] trial increment + count_sessions + CSV columns (110, no rssi)")


def test_pair_delay_columns() -> None:
    """rx1/rx2 timestamp 차이가 있는 pair 저장 후 신규 컬럼 정확성 검증."""
    import pandas as pd

    from collect.recorder import CSV_COLUMNS

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        rec = SessionRecorder(raw_dir=raw)
        rec.start_session("WALK", env=1, subject=1)
        rx1 = CsiPacket(
            DEVICE_ID_RX1, -40, 1, 1_000_000, [float(i) for i in range(AMPLITUDE_COUNT)]
        )
        rx2 = CsiPacket(
            DEVICE_ID_RX2, -45, 1, 1_030_000, [float(i + 100) for i in range(AMPLITUDE_COUNT)]
        )
        rec.add_pair(rx1, rx2)
        buf = rec.stop_session()
        path = rec.save_session(buf, "WALK", env=1, subject=1)

        df = pd.read_csv(path)
        assert len(df.columns) == 110, len(df.columns)
        assert list(df.columns) == CSV_COLUMNS
        r = df.iloc[0]
        # timestamp_us는 Rx1 의미 유지 + timestamp_rx1_us와 동일
        assert int(r["timestamp_us"]) == 1_000_000
        assert int(r["timestamp_rx1_us"]) == 1_000_000
        assert int(r["timestamp_us"]) == int(r["timestamp_rx1_us"])
        assert int(r["timestamp_rx2_us"]) == 1_030_000
        assert int(r["pair_dt_us"]) == abs(1_000_000 - 1_030_000) == 30_000
    print("[OK] pair delay columns (timestamp_rx1/rx2_us, pair_dt_us)")


def test_quality_summary() -> None:
    """summarize_session: 0/1 row, legacy(no pair_dt), 신규 버퍼 안전 동작."""
    from collect.quality import summarize_session

    # 0 row
    s0 = summarize_session([])
    assert s0["pair_count"] == 0
    assert s0["pair_dt_p95_us"] is None and s0["gap_max_us"] is None
    assert s0["loss_rate"] == 0.0

    # 1 row → gap 계산 불가(None), pair_dt는 단일값
    s1 = summarize_session([{"timestamp_us": 100, "seq_rx1": 0, "seq_rx2": 0, "pair_dt_us": 20}])
    assert s1["pair_count"] == 1
    assert s1["gap_p95_us"] is None and s1["gap_max_us"] is None
    assert s1["pair_dt_p50_us"] == 20.0

    # legacy 버퍼 (pair_dt_us 없음) → pair_dt None, gap은 계산됨
    legacy = [
        {"timestamp_us": i * 10_000, "seq_rx1": i, "seq_rx2": i} for i in range(5)
    ]
    sl = summarize_session(legacy)
    assert sl["pair_dt_p95_us"] is None
    assert sl["gap_max_us"] is not None and sl["gap_p95_us"] is not None
    assert sl["pair_rate_hz"] > 0

    # 신규 버퍼 → pair_dt 통계 계산
    new = [
        {"timestamp_us": i * 10_000, "seq_rx1": i, "seq_rx2": i, "pair_dt_us": 1000 + i}
        for i in range(5)
    ]
    sn = summarize_session(new)
    assert sn["pair_dt_max_us"] == 1004.0
    assert sn["pair_dt_p50_us"] is not None
    print("[OK] quality summary (0/1 rows, legacy/new buffers)")


def test_trial_no_overwrite() -> None:
    """T001, T003만 존재 → 다음은 T004 (T003 덮어쓰기 방지)."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        raw.mkdir(parents=True)
        # T002가 비어 있는 상태를 인위적으로 만든다
        (raw / "E1_S01_A_WALK_T001.csv").write_text("dummy\n", encoding="utf-8")
        (raw / "E1_S01_A_WALK_T003.csv").write_text("dummy\n", encoding="utf-8")

        rec = SessionRecorder(raw_dir=raw)
        # count_sessions은 파일 수 그대로 반환 (현황 표시용)
        assert rec.count_sessions("WALK", 1, 1) == 2

        buf = [
            {"timestamp_us": 0, "seq_rx1": 0, "seq_rx2": 0}
            | {f"amp_rx1_{i}": 0.0 for i in range(AMPLITUDE_COUNT)}
            | {f"amp_rx2_{i}": 0.0 for i in range(AMPLITUDE_COUNT)}
        ]
        path = rec.save_session(buf, "WALK", 1, 1)
        assert path.name == "E1_S01_A_WALK_T004.csv", path.name
        # 기존 T003은 그대로 유지
        assert (raw / "E1_S01_A_WALK_T003.csv").read_text(encoding="utf-8") == "dummy\n"

        # 추가 검증: 계산된 경로가 이미 존재해도 빈 번호를 찾는다
        # T004가 방금 저장됐고, 강제로 T005도 외부에서 만들어둔 상태에서
        # _next_trial은 max+1=T006을 줄 텐데 T006은 비어있으니 그대로 저장된다.
        (raw / "E1_S01_A_WALK_T005.csv").write_text("dummy\n", encoding="utf-8")
        path2 = rec.save_session(buf, "WALK", 1, 1)
        assert path2.name == "E1_S01_A_WALK_T006.csv", path2.name

        # 추가 검증: max+1 경로가 비정상적으로 이미 존재할 때 1씩 올려가며 빈 번호 탐색.
        # T010이 외부에서 생성되었지만 T007/T008/T009는 비어있는 상태 — max=T010 → 다음은 T011
        (raw / "E1_S01_A_WALK_T010.csv").write_text("dummy\n", encoding="utf-8")
        path3 = rec.save_session(buf, "WALK", 1, 1)
        assert path3.name == "E1_S01_A_WALK_T011.csv", path3.name
    print("[OK] trial no-overwrite (T003 보존, max+1, collision-safe)")


def test_bind_failure_raises_runtime_error() -> None:
    """동일 포트가 이미 점유되었을 때 start_udp_receiver()는 RuntimeError를 던져야 한다."""
    blocker = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    # Windows에서 동일 포트 wildcard(0.0.0.0) bind 충돌을 재현하려면 blocker도 0.0.0.0으로 bind.
    blocker.bind(("0.0.0.0", 0))
    port = blocker.getsockname()[1]
    # SO_REUSEADDR을 쓰지 않으면 동일 포트 재bind는 OSError
    try:
        try:
            start_udp_receiver(lambda r1, r2: None, port=port)
        except RuntimeError as e:
            msg = str(e)
            assert f"포트 {port}" in msg, msg
            assert "--port" in msg, msg
        else:
            raise AssertionError("RuntimeError가 발생하지 않음")
    finally:
        blocker.close()
    print("[OK] bind failure raises RuntimeError with helpful message")


def test_countdown_has_sleep() -> None:
    """_countdown 함수 구현이 time.sleep을 호출하는지 정적 검증."""
    import inspect

    from collect.collect_main import _countdown

    src = inspect.getsource(_countdown)
    assert "time.sleep" in src, "_countdown must call time.sleep(1) per tick"
    print("[OK] countdown contains time.sleep")


def test_labels() -> None:
    from collect.labels import ACTIVITY_INFO, get_duration, total_target_sessions

    # 240 세션 검증
    assert total_target_sessions() == 240, total_target_sessions()
    # FALL_*: 6종 × target=10 (side fall 제외)
    fall_codes = [c for c, info in ACTIVITY_INFO.items() if info["class_idx"] == 0]
    assert len(fall_codes) == 6
    assert all(ACTIVITY_INFO[c]["target"] == 10 for c in fall_codes)
    assert all(not c.endswith("_S") for c in fall_codes)
    # 비낙상: 6종 × 30
    nonfall = [c for c, info in ACTIVITY_INFO.items() if info["class_idx"] != 0]
    assert len(nonfall) == 6
    assert all(ACTIVITY_INFO[c]["target"] == 30 for c in nonfall)
    # duration 자동 합산
    assert get_duration("FALL_SIT_F") == 5  # 2 + 3
    assert get_duration("WALK") == 8
    assert get_duration("LIE") == 7  # 3 + 4
    print("[OK] labels (240 sessions, side fall excluded, durations)")


def main() -> int:
    test_parse_packet()
    test_pairing()
    test_loss_rate()
    test_trial_and_count()
    test_pair_delay_columns()
    test_quality_summary()
    test_trial_no_overwrite()
    test_bind_failure_raises_runtime_error()
    test_countdown_has_sleep()
    test_labels()
    print("ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
