# server/notification/sms.py

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SOLAPI_API_KEY")
API_SECRET = os.getenv("SOLAPI_API_SECRET")
SENDER = os.getenv("SOLAPI_SENDER")
RECEIVER = os.getenv("SOLAPI_RECEIVER")
SMS_ENABLED = os.getenv("SMS_ENABLED", "true").lower() == "true"

def reload_config():
    global API_KEY, API_SECRET, SENDER, RECEIVER, SMS_ENABLED
    load_dotenv(override=True)
    API_KEY = os.getenv("SOLAPI_API_KEY")
    API_SECRET = os.getenv("SOLAPI_API_SECRET")
    SENDER = os.getenv("SOLAPI_SENDER")
    RECEIVER = os.getenv("SOLAPI_RECEIVER")
    SMS_ENABLED = os.getenv("SMS_ENABLED", "true").lower() == "true"
    print(f"[SMS] 설정 재로드 완료 - SMS_ENABLED={SMS_ENABLED}, RECEIVER={RECEIVER}")

def send_fall_sms():
    if not SMS_ENABLED:
        print("[SMS] SMS 비활성화 상태 - 발송 생략")
        return

    if not RECEIVER:
        print("[SMS] 수신 번호 없음 - 발송 생략")
        return

    try:
        from solapi import SolapiMessageService
        from solapi.model import RequestMessage

        message_service = SolapiMessageService(
            api_key=API_KEY,
            api_secret=API_SECRET
        )

        message = RequestMessage(
            from_=SENDER,
            to=RECEIVER,
            text="[낙상 감지] 어르신의 낙상이 감지되었습니다. 즉시 확인해주세요."
        )

        response = message_service.send(message)
        print(f"[SMS] 보호자에게 문자 발송 완료: {response}")
        try:
            from logger.log_manager import log_info
            log_info(f"[SMS] 발송 완료: {response}")
        except Exception:
            pass

    except Exception as e:
        print(f"[SMS] 오류: {e}")
        try:
            from logger.log_manager import log_warn
            log_warn(f"[SMS] 발송 오류: {e}")
        except Exception:
            pass