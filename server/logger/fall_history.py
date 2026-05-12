# server/logger/fall_history.py

import csv
import os
from datetime import datetime
import threading

# CSV 저장 경로
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(HISTORY_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(HISTORY_DIR, "fall_history.csv")

# CSV 헤더
HEADERS = ["번호", "날짜", "시각", "seq_num", "rx1_서브캐리어", "rx2_서브캐리어", "비고"]

lock = threading.Lock()

def _init_csv():
    """CSV 파일 없으면 헤더 포함해서 생성"""
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

def save_fall(fall_count: int, rx1: dict, rx2: dict, note: str = "자동 감지"):
    """낙상 감지 시 CSV에 기록"""
    _init_csv()
    now = datetime.now()

    row = [
        fall_count,
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        rx1.get("seq_num", "-"),
        rx1.get("n_subcarriers", "-"),
        rx2.get("n_subcarriers", "-"),
        note
    ]

    with lock:
        with open(HISTORY_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    print(f"[HISTORY] 낙상 기록 저장 완료: {HISTORY_FILE}")

def get_history_filepath():
    return HISTORY_FILE