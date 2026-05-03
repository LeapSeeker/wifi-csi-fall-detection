# server/ws_handler/rpi_connection.py

import asyncio
import websockets
import threading

class RPiConnection:
    def __init__(self, on_status_change=None):
        self.websocket = None
        self.lock = asyncio.Lock()
        self.connected = False
        self.on_status_change = on_status_change
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
            # 연결 유지 — Pi4에서 오는 메시지 대기 (재연결 감지용)
            async for message in websocket:
                print(f"[WS] Pi4 수신: {message}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected = False
            self.websocket = None
            print("[WS] Pi4 연결 끊김. 재연결 대기 중...")
            if self.on_status_change:
                self.on_status_change(False)

    def send_fall_alert(self):
        """낙상 감지 시 Pi4에 결과 전송"""
        if not self.connected or self.websocket is None:
            print("[WS] Pi4 미연결 상태 - 알림 전송 실패")
            return
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._send("FALL_DETECTED"),
            self.loop
        )

    async def _send(self, message: str):
        try:
            await self.websocket.send(message)
            print(f"[WS] Pi4에 낙상 알림 전송 완료: {message}")
        except Exception as e:
            print(f"[WS] 전송 오류: {e}")
            self.connected = False
            self.websocket = None