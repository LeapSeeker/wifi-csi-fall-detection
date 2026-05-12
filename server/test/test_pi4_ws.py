# server/test/test_pi4_ws.py

import asyncio
import websockets
import json

# 서버 설정 (노트북 로컬 테스트)
SERVER_IP = "127.0.0.1"
WS_PORT = 8765

async def run():
    uri = f"ws://{SERVER_IP}:{WS_PORT}"
    print(f"[PI4 TEST] 서버에 WebSocket 연결 시도: {uri}")

    async with websockets.connect(uri) as ws:
        print("[PI4 TEST] 연결 성공! 대기 중...")

        # 대시보드에서 Pi4 연결됨 표시 확인
        try:
            async for message in ws:
                print(f"[PI4 TEST] 서버로부터 수신: {message}")

                if message == "FALL_DETECTED":
                    print("[PI4 TEST] ⚠️ 낙상 감지 알림 수신!")
                    print("[PI4 TEST] → 음성 안내 재생 (시뮬레이션)")
                    print("[PI4 TEST] → 경보 취소 버튼 대기 중...")

        except websockets.exceptions.ConnectionClosed:
            print("[PI4 TEST] 연결 끊김")

if __name__ == "__main__":
    asyncio.run(run())