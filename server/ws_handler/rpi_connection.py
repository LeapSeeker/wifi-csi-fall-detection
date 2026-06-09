# server/ws_handler/rpi_connection.py

import asyncio
import datetime
import json
import threading

import websockets

from protocol.pi4_messages import build_fall_alert_payload

class RPiConnection:
    def __init__(self, on_status_change=None, on_rescue_request=None, on_rpi_log=None):
        self.websocket = None
        self.lock = asyncio.Lock()
        self.connected = False
        self.on_status_change = on_status_change
        self.on_rescue_request = on_rescue_request
        self.on_rpi_log = on_rpi_log
        self.loop = None

    def start(self):
        """WebSocket 서버를 별도 스레드에서 실행"""
        t = threading.Thread(target=self._run_server, daemon=True)
        t.start()

    def _run_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self):
        from config.settings import WS_HOST, WS_PORT
        print(f"[WS] Pi4 연결 대기 중... (포트 {WS_PORT})")
        async with websockets.serve(self._handle_client, WS_HOST, WS_PORT):
            await asyncio.Future()  # 서버 상시 유지

    async def _handle_client(self, websocket):
        """Pi4 연결 수락 및 유지"""
        self.websocket = websocket
        self.connected = True
        addr = websocket.remote_address
        print(f"[WS] Pi4 연결됨: {addr}")

        if self.on_status_change:
            self.on_status_change(True)

        try:
            async for message in websocket:
                print(f"[WS] Pi4 수신: {message}")
                try:
                    data = json.loads(message)
                    event = data.get("event")
                    if event == "rescue_request":
                        print("[WS] 구조 요청 수신 — SMS 발송")
                        if self.on_rescue_request:
                            threading.Thread(
                                target=self.on_rescue_request, daemon=True
                            ).start()
                    elif event == "rpi_log":
                        if self.on_rpi_log:
                            self.on_rpi_log(
                                data.get("message", ""),
                                data.get("level", "info"),
                            )
                except (json.JSONDecodeError, Exception) as e:
                    print(f"[WS] 메시지 파싱 오류: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected = False
            self.websocket = None
            print("[WS] Pi4 연결 끊김. 재연결 대기 중...")
            if self.on_status_change:
                self.on_status_change(False)

    def send_fall_alert(
        self,
        confidence: float = 0.0,
        seq_num: int = 0,
        timestamp_us: int = 0,
    ):
        """낙상 감지 시 Pi4에 결과 전송 (JSON 포맷, D-008 + 본 작업 확정).

        포맷은 protocol.pi4_messages.build_fall_alert_payload()를 기준으로 한다.
        """
        if not self.connected or self.websocket is None:
            print("[WS] Pi4 미연결 상태 - 알림 전송 실패")
            return
        if self.loop is None:
            return

        if not timestamp_us:
            timestamp_us = int(datetime.datetime.now().timestamp() * 1_000_000)

        payload = build_fall_alert_payload(
            confidence=confidence,
            seq_num=seq_num,
            timestamp_us=timestamp_us,
        )
        asyncio.run_coroutine_threadsafe(
            self._send(json.dumps(payload)),
            self.loop,
        )

    async def _send(self, message: str):
        try:
            await self.websocket.send(message)
            print(f"[WS] Pi4에 낙상 알림 전송 완료: {message}")
        except Exception as e:
            print(f"[WS] 전송 오류: {e}")
            self.connected = False
            self.websocket = None
