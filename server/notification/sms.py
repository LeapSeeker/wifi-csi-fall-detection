# server/notification/sms.py

import hashlib
import hmac
import time
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SOLAPI_API_KEY")
API_SECRET = os.getenv("SOLAPI_API_SECRET")
SENDER = os.getenv("SOLAPI_SENDER")
RECEIVER = os.getenv("SOLAPI_RECEIVER")

def _get_auth_header() -> dict:
    date = str(int(time.time() * 1000))
    salt = date
    combined = date + salt
    signature = hmac.new(
        API_SECRET.encode(),
        combined.encode(),
        hashlib.sha256
    ).hexdigest()

    return {
        "Authorization": f"HMAC-SHA256 apiKey={API_KEY}, date={date}, salt={salt}, signature={signature}",
        "Content-Type": "application/json"
    }

def send_fall_sms():
    url = "https://api.solapi.com/messages/v4/send"
    headers = _get_auth_header()
    payload = {
        "message": {
            "to": RECEIVER,
            "from": SENDER,
            "text": "[낙상 감지] 어르신의 낙상이 감지되었습니다. 즉시 확인해주세요."
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print("[SMS] 보호자에게 문자 발송 완료")
        else:
            print(f"[SMS] 발송 실패: {response.status_code} {response.text}")
    except Exception as e:
        print(f"[SMS] 오류: {e}")