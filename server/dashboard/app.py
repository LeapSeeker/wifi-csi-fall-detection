# server/dashboard/app.py

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import datetime
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------------------------
# 서버 상태 저장
# -----------------------------------------------
state = {
    "rpi4_connected": False,
    "pair_count": 0,
    "fall_count": 0,
    "last_fall_time": None,
    "recent_pairs": [],
    "packet_stats": {
        "rx1": {"received": 0, "lost": 0, "loss_rate": 0.0},
        "rx2": {"received": 0, "lost": 0, "loss_rate": 0.0}
    }
}

# -----------------------------------------------
# 상태 업데이트 함수
# -----------------------------------------------
import time

# 초당 패킷 카운터
_packet_count_window = []
_packet_count_lock = threading.Lock()

def update_pair(rx1, rx2):
    state["pair_count"] += 1

    # 초당 패킷 수신 카운트
    now = time.time()
    with _packet_count_lock:
        _packet_count_window.append(now)
        # 최근 1초 이내 것만 유지
        cutoff = now - 1.0
        while _packet_count_window and _packet_count_window[0] < cutoff:
            _packet_count_window.pop(0)
        pps = len(_packet_count_window)  # packets per second

    record = {
        "seq": rx1["seq_num"],
        "rx1_subs": rx1["n_subcarriers"],
        "rx2_subs": rx2["n_subcarriers"],
        "rx1_ts": rx1["timestamp_us"],
        "rx2_ts": rx2["timestamp_us"],
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "pps": pps
    }
    state["recent_pairs"].insert(0, record)
    state["recent_pairs"] = state["recent_pairs"][:10]
    socketio.emit("pair_update", record)

def update_fall():
    state["fall_count"] += 1
    state["last_fall_time"] = datetime.datetime.now().strftime("%H:%M:%S")
    socketio.emit("fall_detected", {
        "fall_count": state["fall_count"],
        "time": state["last_fall_time"]
    })

import time as _time

_rpi4_connected_time = None

def update_rpi4_status(connected: bool):
    global _rpi4_connected_time
    state["rpi4_connected"] = connected

    if connected:
        _rpi4_connected_time = _time.time()
        socketio.emit("rpi4_status", {"connected": True, "message": "WebSocket 연결 활성"})
    else:
        duration = "-"
        if _rpi4_connected_time:
            elapsed = int(_time.time() - _rpi4_connected_time)
            duration = f"{elapsed}초 연결 유지 후 끊김"
        socketio.emit("rpi4_status", {"connected": False, "message": duration})
        socketio.emit("rpi4_disconnected", {"message": duration})

def update_packet_stats(stats: dict):
    state["packet_stats"] = stats
    socketio.emit("packet_stats", stats)

# -----------------------------------------------
# 라우트
# -----------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify(state)

@app.route("/trigger_fall", methods=["POST"])
def trigger_fall():
    update_fall()
    return jsonify({"status": "ok"})

@app.route("/health")
def health():
    from datetime import datetime
    return jsonify({
        "status": "ok",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rpi4_connected": state["rpi4_connected"],
        "pair_count": state["pair_count"],
        "fall_count": state["fall_count"],
        "packet_stats": state["packet_stats"]
    })

# -----------------------------------------------
# 대시보드 실행
# -----------------------------------------------
def start_dashboard():
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)