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

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    print("[PI4 TEST] JSON 파싱 실패 - 메시지 무시")
                    continue

                if data.get("event") == "fall_detected" and data.get("label") == "fall":
                    confidence = data.get("confidence", 0.0)
                    seq_num = data.get("seq_num", 0)
                    timestamp_us = data.get("timestamp_us", 0)
                    print(
                        f"[PI4 TEST] ⚠️ 낙상 감지 알림 수신! "
                        f"confidence={confidence:.3f} seq={seq_num} ts={timestamp_us}"
                    )
                    print("[PI4 TEST] → 음성 안내 재생 (시뮬레이션)")
                    print("[PI4 TEST] → 경보 취소 버튼 대기 중...")

        except websockets.exceptions.ConnectionClosed:
            print("[PI4 TEST] 연결 끊김")

if __name__ == "__main__":
    asyncio.run(run())