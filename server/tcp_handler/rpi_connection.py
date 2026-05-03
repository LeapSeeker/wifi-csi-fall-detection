# server/tcp_handler/rpi_connection.py

import socket
import threading
from config.settings import TCP_HOST, TCP_PORT

class RPiConnection:
    def __init__(self, on_status_change=None):
        self.client_socket = None
        self.lock = threading.Lock()
        self.connected = False
        self.on_status_change = on_status_change  # 대시보드 상태 업데이트 콜백

    def start(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((TCP_HOST, TCP_PORT))
        server_sock.listen(1)
        print(f"[TCP] RPi4 연결 대기 중... (포트 {TCP_PORT})")

        t = threading.Thread(target=self._accept_loop, args=(server_sock,), daemon=True)
        t.start()

    def _accept_loop(self, server_sock):
        while True:
            try:
                conn, addr = server_sock.accept()
                with self.lock:
                    self.client_socket = conn
                    self.connected = True
                print(f"[TCP] RPi4 연결됨: {addr}")
                if self.on_status_change:
                    self.on_status_change(True)
                self._keep_alive(conn)
            except Exception as e:
                print(f"[TCP] 연결 오류: {e}")

    def _keep_alive(self, conn):
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    raise ConnectionResetError
            except:
                with self.lock:
                    self.connected = False
                    self.client_socket = None
                print("[TCP] RPi4 연결 끊김. 재연결 대기 중...")
                if self.on_status_change:
                    self.on_status_change(False)
                break

    def send_fall_alert(self):
        with self.lock:
            if not self.connected or self.client_socket is None:
                print("[TCP] RPi4 미연결 상태 - 알림 전송 실패")
                return
            try:
                self.client_socket.sendall(b"FALL_DETECTED")
                print("[TCP] RPi4에 낙상 알림 전송 완료")
            except Exception as e:
                print(f"[TCP] 전송 오류: {e}")
                self.connected = False
                self.client_socket = None