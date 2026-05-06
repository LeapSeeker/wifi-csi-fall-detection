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
SMS_ENABLED = os.getenv("SMS_ENABLED", "true").lower() == "true"

def reload_config():
    """설정 변경 시 즉시 반영"""
    global API_KEY, API_SECRET, SENDER, RECEIVER, SMS_ENABLED
    load_dotenv(override=True)
    API_KEY = os.getenv("SOLAPI_API_KEY")
    API_SECRET = os.getenv("SOLAPI_API_SECRET")
    SENDER = os.getenv("SOLAPI_SENDER")
    RECEIVER = os.getenv("SOLAPI_RECEIVER")
    SMS_ENABLED = os.getenv("SMS_ENABLED", "true").lower() == "true"
    print(f"[SMS] 설정 재로드 완료 — SMS_ENABLED={SMS_ENABLED}, RECEIVER={RECEIVER}")

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
    if not SMS_ENABLED:
        print("[SMS] SMS 비활성화 상태 — 발송 생략")
        return

    if not RECEIVER:
        print("[SMS] 수신 번호 없음 — 발송 생략")
        return

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