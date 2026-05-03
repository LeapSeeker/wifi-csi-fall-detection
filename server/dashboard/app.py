# server/dashboard/app.py

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import datetime

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
    "recent_pairs": []
}

# -----------------------------------------------
# 상태 업데이트 함수 (main.py에서 호출)
# -----------------------------------------------
def update_pair(rx1, rx2):
    state["pair_count"] += 1
    record = {
        "seq": rx1["seq_num"],
        "rx1_subs": rx1["n_subcarriers"],
        "rx2_subs": rx2["n_subcarriers"],
        "rx1_ts": rx1["timestamp_us"],
        "rx2_ts": rx2["timestamp_us"],
        "time": datetime.datetime.now().strftime("%H:%M:%S")
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

def update_rpi4_status(connected: bool):
    state["rpi4_connected"] = connected
    socketio.emit("rpi4_status", {"connected": connected})

# -----------------------------------------------
# 라우트
# -----------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify(state)

# -----------------------------------------------
# 대시보드 실행
# -----------------------------------------------
def start_dashboard():
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)