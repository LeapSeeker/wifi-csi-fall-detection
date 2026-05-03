# server/tcp_handler/rpi_connection.py

import socket
import threading
from config.settings import TCP_HOST, TCP_PORT

class RPiConnection:
    def __init__(self):
        self.client_socket = None  # RPi4 연결 소켓
        self.lock = threading.Lock()
        self.connected = False

    def start(self):
        """RPi4의 연결을 기다리는 TCP 서버 시작"""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((TCP_HOST, TCP_PORT))
        server_sock.listen(1)
        print(f"[TCP] RPi4 연결 대기 중... (포트 {TCP_PORT})")

        t = threading.Thread(target=self._accept_loop, args=(server_sock,), daemon=True)
        t.start()

    def _accept_loop(self, server_sock):
        """RPi4 연결 수락 루프 - 끊기면 재연결 대기"""
        while True:
            try:
                conn, addr = server_sock.accept()
                with self.lock:
                    self.client_socket = conn
                    self.connected = True
                print(f"[TCP] RPi4 연결됨: {addr}")

                # 연결 유지 감지 (끊기면 다시 대기)
                self._keep_alive(conn)

            except Exception as e:
                print(f"[TCP] 연결 오류: {e}")

    def _keep_alive(self, conn):
        """연결 끊김 감지"""
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
                break

    def send_fall_alert(self):
        """낙상 감지 시 RPi4에 알림 전송"""
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