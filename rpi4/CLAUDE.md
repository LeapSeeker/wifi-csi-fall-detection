# rpi4/CLAUDE.md — Raspberry Pi 4 파이프라인
# 담당: 동석 (통신/알림), 주화 (환경 설정/오디오)
# 브랜치: feature/rpi4-pipeline
# 공유 인터페이스 → context/SHARED.md

## 하드웨어

| 장치 | 역할 |
|------|------|
| Raspberry Pi 4 | 메인 I/O 허브 |
| ReSpeaker Lite USB 마이크 어레이 | 오디오 출력 |
| Seeed Studio 스피커 (JST) | 음성 출력 |
| 하드웨어 버튼 × 2 | ① 경보 취소 / ② 긴급 SMS |

- hostname: safesignal
- 전원: 유선 어댑터 (잠정 합의)

## 데이터 흐름

추론 서버 (WebSocket 서버)
↑ (Pi4 아웃바운드 연결)
Raspberry Pi 4
├─ 낙상 감지 수신 → 오디오 파일 재생
├─ 버튼 ① 입력 → 경보 취소
└─ 버튼 ② 입력 → SOLAPI SMS 전송
추론 결과 포맷 → `context/SHARED.md` 참조

## 오디오

- TTS는 사전 생성된 음성 파일만 사용 (런타임 TTS 없음)
- 파일 재생 방식으로 구현

## SMS 알림 (SOLAPI)

- 담당: 동석
- 서비스: SOLAPI (구 CoolSMS)
- 트리거: 낙상 감지 자동 발송 + 버튼 ② 수동 발송
- 목표 응답: ≤ 1초
- 공식 문서: https://docs.solapi.com

## 참고 자료

- SOLAPI Python SDK: https://github.com/solapi/solapi-python

## TODO

- [ ] 자동/수동 발송 세부 시나리오 팀 합의 필요 (동석)
- [ ] Pi4 전원 방식 최종 확정