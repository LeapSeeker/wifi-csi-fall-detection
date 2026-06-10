# rpi_receiver.py

import asyncio
import websockets
import json
import subprocess
import threading
import time
import os
import RPi.GPIO as GPIO
from gtts import gTTS
import pygame

# ── 설정 ──────────────────────────────────────────────
SERVER_IP    = os.environ.get("SERVER_IP", "192.168.0.5")
SERVER_PORT  = int(os.environ.get("SERVER_PORT", "8765"))
CANCEL_PIN   = 17   # 버튼1: 낙상 감지 취소 (안전확인)
RESCUE_PIN   = 27   # 버튼2: 구조 요청
CANCEL_SEC   = 10
TTS_PATH     = "/tmp/tts_output.mp3"
# ──────────────────────────────────────────────────────

# ── GPIO 초기화 ───────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setup(CANCEL_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(RESCUE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

cancel_flag = False          # 낙상 감지 취소 플래그
rescue_cancel_flag = False   # 구조 요청 취소 플래그
rescue_in_progress = False   # 구조 요청 대기 중 여부

_ws_ref = None
_loop_ref = None

# ── 서버로 로그 전송 ──────────────────────────────────
def send_log(msg: str, level: str = "info") -> None:
    """print + 서버 대시보드로 로그 전송."""
    print(msg)
    if _ws_ref and _loop_ref:
        try:
            asyncio.run_coroutine_threadsafe(
                _ws_ref.send(json.dumps({
                    "event": "rpi_log",
                    "message": msg,
                    "level": level,
                })),
                _loop_ref,
            )
        except Exception:
            pass

# ── pygame 초기화 ─────────────────────────────────────
_pygame_ok = False
try:
    pygame.mixer.pre_init(frequency=48000, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    _pygame_ok = True
except Exception as e:
    print(f"[pygame] 오디오 초기화 실패: {e} — espeak-ng로 대체")

# ── 버튼 리스너 ───────────────────────────────────────
def cancel_button_listener():
    """버튼1: 낙상 감지 취소 — 눌리는 순간 1회만 감지."""
    global cancel_flag
    prev_pressed = False
    while True:
        is_pressed = GPIO.input(CANCEL_PIN) == GPIO.LOW
        if is_pressed and not prev_pressed:
            cancel_flag = True
            send_log("[버튼1] 취소 버튼 눌림")
        prev_pressed = is_pressed
        time.sleep(0.05)


def rescue_button_listener():
    """버튼2: 구조 요청 시작 / 대기 중 재입력 시 취소 — 눌리는 순간 1회만 감지."""
    global rescue_cancel_flag
    prev_pressed = False
    while True:
        is_pressed = GPIO.input(RESCUE_PIN) == GPIO.LOW
        if is_pressed and not prev_pressed:
            if rescue_in_progress:
                # 대기 중 → 버튼2 다시 누르면 취소
                rescue_cancel_flag = True
                send_log("[버튼2] 구조 요청 취소")
            else:
                # 대기 중 아님 → 구조 요청 시작
                send_log("[버튼2] 구조 요청 버튼 눌림", level="warn")
                threading.Thread(target=handle_rescue, daemon=True).start()
        prev_pressed = is_pressed
        time.sleep(0.05)

# ── 음성 출력 (gTTS → espeak-ng 순서로 시도) ──────────
def speak(text):
    print(f"[TTS] 발화 시작: {text[:20]}...")
    # 1차: gTTS + pygame (오디오 장치 사용 가능할 때만)
    if _pygame_ok:
        try:
            tts = gTTS(text=text, lang='ko')
            tts.save(TTS_PATH)
            pygame.mixer.music.load(TTS_PATH)
            pygame.mixer.music.set_volume(1.0)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            print("[TTS] 완료 (gTTS)")
            return
        except Exception as e:
            print(f"[TTS] gTTS 오류: {e}")

    # 2차: espeak-ng (blocking)
    try:
        subprocess.run(
            ['espeak-ng', '-v', 'ko', '-s', '130', text],
            timeout=15
        )
        print("[TTS] 완료 (espeak-ng)")
    except FileNotFoundError:
        print("[TTS] espeak-ng 없음 — 소리 출력 불가")
    except Exception as e:
        print(f"[TTS] espeak-ng 오류: {e}")

# ── 낙상 감지 처리 (서버→RPi) ─────────────────────────
def handle_fall(data):
    global cancel_flag
    cancel_flag = False

    confidence = data.get("confidence", 0.0)
    seq_num    = data.get("seq_num", 0)

    send_log("[낙상 감지] 낙상 감지됨", level="warn")

    # 1. 즉시 음성 안내
    speak("낙상이 감지되었습니다. 괜찮으시면 버튼을 눌러주세요.")

    # 2. 10초 대기 — 버튼1 누르면 취소
    print(f"[대기] {CANCEL_SEC}초 내 버튼 누르면 취소...")
    for _ in range(CANCEL_SEC * 10):
        if cancel_flag:
            send_log("[취소] 오탐으로 처리 — SMS 생략")
            if _ws_ref and _loop_ref:
                asyncio.run_coroutine_threadsafe(
                    _ws_ref.send(json.dumps({"event": "fall_cancel"})),
                    _loop_ref
                )
            speak("취소되었습니다.")
            return
        time.sleep(0.1)

    # 3. 버튼 안 눌리면 서버에 SMS 요청 후 음성 재안내
    send_log("[낙상 감지] 보호자 SMS 전송 요청", level="warn")
    if _ws_ref and _loop_ref:
        asyncio.run_coroutine_threadsafe(
            _ws_ref.send(json.dumps({"event": "rescue_request"})),
            _loop_ref,
        )
    speak("보호자에게 연락합니다.")

# ── 구조 요청 처리 (RPi→서버) ─────────────────────────
def handle_rescue():
    global rescue_cancel_flag, rescue_in_progress
    rescue_cancel_flag = False
    rescue_in_progress = True

    send_log("[구조 요청] 처리 시작", level="warn")

    # 1. 음성 안내
    speak("구조 요청이 감지되었습니다. 보호자에게 연락합니다. 취소를 원하시면 버튼을 눌러주세요.")

    # 2. 10초 대기 — 버튼2 다시 누르면 취소
    for _ in range(CANCEL_SEC * 10):
        if rescue_cancel_flag:
            send_log("[구조 취소] 구조 요청 취소됨")
            speak("취소되었습니다.")
            rescue_in_progress = False
            return
        time.sleep(0.1)

    rescue_in_progress = False

    # 3. 취소 안 됐으면 서버로 구조 요청 전송 → SMS 발송
    send_log("[구조 요청] 서버에 전송 중...", level="warn")
    if _ws_ref and _loop_ref:
        asyncio.run_coroutine_threadsafe(
            _ws_ref.send(json.dumps({"event": "rescue_request"})),
            _loop_ref
        )
    send_log("[구조 요청] 보호자 SMS 전송 완료", level="warn")
    speak("보호자에게 연락했습니다.")

# ── WebSocket 클라이언트 ──────────────────────────────
async def connect_to_server():
    global _ws_ref, _loop_ref
    uri = f"ws://{SERVER_IP}:{SERVER_PORT}"

    while True:
        try:
            print(f"[WS] 서버 연결 시도 중... {uri}")
            async with websockets.connect(uri) as ws:
                _ws_ref = ws
                _loop_ref = asyncio.get_event_loop()
                print(f"[WS] 서버 연결 완료")

                async for message in ws:
                    try:
                        data = json.loads(message)
                        print(f"[WS] 수신: {data}")

                        if data.get("event") == "fall_detected":
                            t = threading.Thread(
                                target=handle_fall, args=(data,), daemon=True
                            )
                            t.start()

                    except json.JSONDecodeError:
                        print(f"[WS] JSON 파싱 오류: {message}")

        except Exception as e:
            print(f"[WS] 연결 실패: {e} → 5초 후 재연결")
            await asyncio.sleep(5)

# ── 메인 ─────────────────────────────────────────────
if __name__ == "__main__":
    # 버튼 리스너 스레드 시작
    threading.Thread(target=cancel_button_listener, daemon=True).start()
    threading.Thread(target=rescue_button_listener, daemon=True).start()

    print("[시작] 라즈베리파이 낙상 감지 수신기")
    print(f"  서버: {SERVER_IP}:{SERVER_PORT}")
    print(f"  버튼1 (취소) GPIO: {CANCEL_PIN}")
    print(f"  버튼2 (구조 요청) GPIO: {RESCUE_PIN}")
    print(f"  취소 시간: {CANCEL_SEC}초")

    try:
        asyncio.run(connect_to_server())
    except KeyboardInterrupt:
        print("\n[종료]")
        GPIO.cleanup()
        pygame.mixer.quit()
