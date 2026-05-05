# server/logger/log_manager.py

import logging
import os
from datetime import datetime

# 로그 폴더 생성
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 로그 파일명 (서버 시작 시각 기준)
log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_server.log"
log_filepath = os.path.join(LOG_DIR, log_filename)

# 로거 설정
logger = logging.getLogger("fall_detection")
logger.setLevel(logging.DEBUG)

# 파일 핸들러
file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

# 터미널 핸들러
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# 포맷
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def log_info(msg): logger.info(msg)
def log_warn(msg): logger.warning(msg)
def log_error(msg): logger.error(msg)
def log_debug(msg): logger.debug(msg)

def log_pair(rx1, rx2):
    logger.info(
        f"[PAIR] seq={rx1['seq_num']} | "
        f"RX1 subs={rx1['n_subcarriers']} | "
        f"RX2 subs={rx2['n_subcarriers']}"
    )

def log_fall(fall_count: int):
    logger.warning(
        f"[FALL] 낙상 감지! 총 {fall_count}회 | "
        f"시각={datetime.now().strftime('%H:%M:%S')}"
    )

def get_log_filepath():
    return log_filepath