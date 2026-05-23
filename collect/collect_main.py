"""SafeSignal 자체 데이터 수집 — CLI 진입점.

실행:
    python collect/collect_main.py [--port 5005]

Google Drive 자동 업로드(rclone 사용, 선택):
    $env:SAFESIGNAL_DRIVE_UPLOAD="1"
    $env:SAFESIGNAL_DRIVE_REMOTE="gdrive:SafeSignal/data/raw"

수집 목표 (242 세션):
- 낙상 6종 × 10회 = 60 세션
- 비낙상 6종 × 30회 = 180 세션
- no_motion baseline 2회
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# `python collect/collect_main.py` 실행 시 패키지 import 가능하도록 루트 추가
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from collect.beep import beep_end, beep_ready, beep_stage
from collect.drive_upload import DriveUploadConfig, upload_file_async, upload_status_message
from collect.labels import ACTIVITY_INFO, ACTIVITY_ORDER, get_duration
from collect.quality import format_quality_lines, summarize_session
from collect.recorder import SessionRecorder
from collect.udp import UDP_PORT, start_udp_receiver

# ── 출력 헬퍼 ────────────────────────────────────────────
GUIDE_TEXT = """
[수집 가이드라인]
- 각자의 자연스러운 자세로 수행하세요.
- 의도적인 자세 변형(허리 굽힘, 무릎 굽힘 등)은 금지합니다.
- 메트 위에서 수집을 진행하세요 (안전).
"""

PORT_NOTICE = """
[포트 안내]
기본 UDP 포트는 5005입니다.
Windows에서는 같은 UDP 포트를 다른 서버와 동시에 안정적으로 공유하지 않습니다.
server/dongseok 수신 서버와 collect를 동시에 실행하지 마세요.
"""


def _read_int(prompt: str) -> int:
    while True:
        s = input(prompt).strip()
        try:
            return int(s)
        except ValueError:
            print("[ERR] 숫자를 입력하세요.")


def _print_status_table(recorder: SessionRecorder, env: int, subject: int) -> None:
    print()
    print(f"=== 활동 코드 현황 (E{env}, S{subject:02d}) ===")
    print(
        f"{'No':>3}  {'코드':<12}  {'표시명':<18}  "
        f"{'완료':>5}  {'목표':>5}  진행"
    )
    for i, code in enumerate(ACTIVITY_ORDER, start=1):
        info = ACTIVITY_INFO[code]
        target = info["target"]
        done = recorder.count_sessions(code, env, subject)
        ratio = min(done / target, 1.0) if target else 0.0
        bar_w = 20
        filled = int(ratio * bar_w)
        bar = "[" + "#" * filled + "." * (bar_w - filled) + "]"
        print(
            f"{i:>3}  {code:<12}  {info['display']:<18}  "
            f"{done:>5}  {target:>5}  {bar} {ratio*100:5.1f}%"
        )
    print()


def _select_activity(recorder: SessionRecorder, env: int, subject: int) -> str:
    while True:
        _print_status_table(recorder, env, subject)
        s = input("활동 번호를 입력하세요: ").strip()
        try:
            n = int(s)
        except ValueError:
            print("[ERR] 숫자를 입력하세요.")
            continue
        if 1 <= n <= len(ACTIVITY_ORDER):
            return ACTIVITY_ORDER[n - 1]
        print(f"[ERR] 1~{len(ACTIVITY_ORDER)} 사이 번호를 입력하세요.")


def _countdown(seconds: int) -> None:
    for i in range(seconds, 0, -1):
        print(f"{i}...", flush=True)
        time.sleep(1)


def _run_session(recorder: SessionRecorder, activity_code: str, env: int, subject: int) -> None:
    info = ACTIVITY_INFO[activity_code]

    # 1. ENTER 후 ready beep
    beep_ready()

    # 2. 3초 카운트다운 (CSV에 포함되지 않음)
    _countdown(3)

    # 3. 카운트다운 종료 직후 start_session
    recorder.start_session(activity_code, env, subject)

    if "stages" in info:
        for i, stage in enumerate(info["stages"]):
            print(f"  [Stage {i + 1}] {stage['name']} ({stage['duration']}s)")
            beep_stage()
            time.sleep(stage["duration"])
    else:
        duration = info["duration"]
        print(f"  [녹화] {info['display']} ({duration}s)")
        beep_stage()
        time.sleep(duration)

    # 마지막 stage/duration 종료 직후 stop_session, 그 다음 end beep
    buf = recorder.stop_session()
    beep_end()

    # 결과 출력
    n_pairs = len(buf)
    loss = recorder.calculate_loss_rate(buf)
    duration_s = 0.0
    pair_rate = 0.0
    if n_pairs >= 2:
        ts_vals = [row["timestamp_us"] for row in buf]
        duration_s = (max(ts_vals) - min(ts_vals)) / 1e6
        if duration_s > 0:
            pair_rate = n_pairs / duration_s
    capture_ratio = pair_rate / 100.0
    print(
        f"  → 페어 수: {n_pairs}, duration: {duration_s:.2f}s, "
        f"실측 rate: {pair_rate:.1f} Hz "
        f"(target≈100, capture_ratio={capture_ratio:.2f}), "
        f"손실률: {loss*100:.1f}%"
    )
    if loss > 0.05:
        print(f"  [경고] 패킷 손실률 {loss*100:.1f}% — 저장 여부를 신중히 결정하세요.")

    # 페어링 품질 요약 (report-only — 저장 차단 조건 아님)
    for line in format_quality_lines(summarize_session(buf)):
        print(line)

    while True:
        ans = input("저장하시겠습니까? (Y/N): ").strip().lower()
        if ans in ("y", "yes"):
            path = recorder.save_session(buf, activity_code, env, subject)
            if path is not None:
                print(f"  저장됨: {path}")
                upload_file_async(path)
            break
        if ans in ("n", "no"):
            print("  폐기됨.")
            break
        print("[ERR] Y 또는 N을 입력하세요.")


def _ask_change() -> str:
    print()
    print("1. 환경 변경")
    print("2. 활동 코드 변경")
    print("3. 피험자 변경")
    print("4. 변경 없음")
    print("q. 종료")
    while True:
        s = input("선택: ").strip().lower()
        if s in ("1", "2", "3", "4", "q"):
            return s
        print("[ERR] 1/2/3/4/q 중 입력하세요.")


def main() -> int:
    parser = argparse.ArgumentParser(description="SafeSignal 자체 데이터 수집")
    parser.add_argument("--port", type=int, default=UDP_PORT, help="UDP 수신 포트")
    args = parser.parse_args()

    print(GUIDE_TEXT)
    print(PORT_NOTICE)
    upload_config = DriveUploadConfig.from_env()
    print(upload_status_message(upload_config))

    recorder = SessionRecorder()

    # UDP receiver: 프로그램 시작 시 1회 실행
    try:
        start_udp_receiver(recorder.add_pair, port=args.port)
    except RuntimeError as e:
        print(f"[ERR] {e}")
        return 1
    print(f"[INFO] UDP 수신 시작 (port={args.port})")

    # 초기 선택 1회 — env → subject → activity 순 (활동 표 완료 수가 정확하려면 subject 먼저 필요)
    env = _read_int("환경 번호 (E?)를 입력하세요: ")
    subject = _read_int("피험자 번호 (S?)를 입력하세요: ")
    activity_code = _select_activity(recorder, env, subject)

    while True:
        info = ACTIVITY_INFO[activity_code]
        done = recorder.count_sessions(activity_code, env, subject)
        target = info["target"]
        duration = get_duration(activity_code)
        print()
        print(
            f"[현재 선택] E{env} / S{subject:02d} / {activity_code} "
            f"({info['display']}, {duration}s) — {done}/{target} 완료"
        )

        s = input(
            "피험자를 준비시킨 후 ENTER를 눌러 녹화를 시작하세요. q 입력 시 종료. "
        ).strip().lower()
        if s == "q":
            print("[INFO] 종료")
            return 0

        _run_session(recorder, activity_code, env, subject)

        choice = _ask_change()
        if choice == "q":
            print("[INFO] 종료")
            return 0
        if choice == "1":
            env = _read_int("환경 번호 (E?)를 입력하세요: ")
        elif choice == "2":
            activity_code = _select_activity(recorder, env, subject)
        elif choice == "3":
            subject = _read_int("피험자 번호 (S?)를 입력하세요: ")
        # choice == "4" → 그대로


if __name__ == "__main__":
    sys.exit(main())
