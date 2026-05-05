# server/test/trigger_fall.py

import asyncio
import websockets
import requests
import time

# 서버 설정
SERVER_IP = "127.0.0.1"
WS_PORT = 8765
DASHBOARD_URL = f"http://{SERVER_IP}:8080"

async def trigger_fall_via_ws():
    """WebSocket으로 직접 낙상 트리거 (디버그용)"""
    uri = f"ws://{SERVER_IP}:{WS_PORT}"
    try:
        async with websockets.connect(uri) as ws:
            print("[TRIGGER] WebSocket 연결됨")
            await ws.send("FALL_DETECTED")
            print("[TRIGGER] FALL_DETECTED 전송 완료")
    except Exception as e:
        print(f"[TRIGGER] 오류: {e}")

def trigger_fall_via_api():
    """대시보드 API로 낙상 트리거"""
    try:
        response = requests.post(f"{DASHBOARD_URL}/trigger_fall")
        if response.status_code == 200:
            print("[TRIGGER] 낙상 감지 시뮬레이션 완료")
        else:
            print(f"[TRIGGER] 실패: {response.status_code}")
    except Exception as e:
        print(f"[TRIGGER] 오류: {e}")

if __name__ == "__main__":
    print("=" * 40)
    print("  낙상 감지 시뮬레이션 트리거")
    print("=" * 40)
    print("1. 대시보드 API로 트리거 (권장)")
    print("2. 종료")
    print()

    choice = input("선택: ").strip()

    if choice == "1":
        print("[TRIGGER] 낙상 감지 시뮬레이션 시작...")
        trigger_fall_via_api()
    else:
        print("[TRIGGER] 종료")