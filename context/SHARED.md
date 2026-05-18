# SHARED.md — 파트 간 공유 인터페이스

이 파일을 수정할 경우 반드시 관련 파트 담당자에게 공유 후 진행.

## 네트워크 설정

| 항목 | 값 |
|------|----|
| UDP 포트 | 5005 |
| 통신 방향 | ESP32-S3 Rx → 추론 서버 (UDP) |
| 추론 서버 → Pi4 | WebSocket (Pi4 아웃바운드 연결) |
| IP 설정 | 고정 필수 (DHCP 예약 또는 정적 설정) |
| 패킷 손실 허용 | ≤ 5% (서버에서 0패딩/선형보간 보완) |

## UDP 패킷 구조
`magic(1B,0xAB) | device_id(1B) | rssi(1B,int8_t) | reserved(1B) | seq(4B) | timestamp_us(8B) | amplitude × 52(208B,float32)`

총 224B

| 필드 | 타입 | 크기 | 값 |
|------|------|------|----|
| magic | uint8 | 1B | 0xAB |
| device_id | uint8 | 1B | Rx1=0x01, Rx2=0x02 |
| rssi | int8 | 1B | 수신 신호 세기(dBm), 전처리/학습 입력에는 사용하지 않음 |
| reserved | uint8 | 1B | 0x00 |
| seq | uint32 | 4B | Rx 장치별 로컬 시퀀스, 패킷마다 +1 |
| timestamp_us | uint64 | 8B | SNTP 기준 Unix time (μs) |
| amplitude | float32 × 52 | 208B | √(I²+Q²) |

> **주의:** 서버의 Rx1/Rx2 페어링은 `seq`가 아니라 `timestamp_us` 기준 nearest match로 수행한다.
> `seq`는 각 Rx 장치별 로컬 카운터이므로 한쪽 패킷 손실 시 서로 어긋날 수 있다.
>
> **주의:** timestamp_us는 SNTP 동기화된 Unix time 기준이어야 함.
> 현재 테스트에서 `time=1777521566s`로 찍히는 값은 2026년 Unix time(약 1746000000s)과 불일치.
> ESP 담당자에게 SNTP 동기화 실제 적용 여부 확인 필요.

## 추론 결과 포맷 (서버 → Pi4, WebSocket)

현재 Pi4 WebSocket 메시지는 **낙상 알림 전용**이다. 서버는 낙상으로 판정된 경우에만
아래 JSON을 Pi4로 전송한다.

```json
{
  "event": "fall_detected",
  "label": "fall",
  "confidence": 0.94,
  "seq_num": 1234,
  "timestamp_us": 1234567890
}
```

| 필드 | 타입 | 값 |
|------|------|----|
| event | string | `fall_detected` |
| label | string | 현재 Pi4 알림 이벤트는 `fall`만 전송 |
| confidence | float | 모델 softmax fall 확률 (0.0~1.0) |
| seq_num | int | 감지 윈도우의 기준 Rx1 패킷 seq |
| timestamp_us | int | 감지 윈도우의 기준 Rx1 패킷 timestamp_us |

> 낙상 3방향(fall_forward/backward/side)은 학습 내부 레이블. 서버 → Pi4 알림 출력은 `fall`로 통합.
> `class` 키는 사용하지 않고 `label`을 사용한다.
> `timestamp_us`는 낙상으로 판정된 윈도우의 기준 Rx1 패킷 timestamp_us를 사용한다.
> 값이 0이거나 누락된 경우에만 서버 현재 시각(Unix μs)을 fallback으로 사용한다.
> 서버 코드의 실제 payload 생성 기준은 `server/protocol/pi4_messages.py`이다.

## 일반 행동 추론 결과 처리

`stand`, `walk`, `sit`, `lying` 등 낙상이 아닌 일반 행동 추론 결과는 현재 Pi4로
실시간 전송하지 않는다. 향후 1주일 단위 생활 패턴 분석에 활용하기 위해 서버
내부 저장 대상으로 분리한다.

| 항목 | 현재 결정 |
|------|-----------|
| Pi4 실시간 전송 | 낙상 알림만 전송 |
| 일반 행동 결과 | 서버 저장 후보 |
| 분석 단위 | 1주일 단위 생활 패턴 변화 |
| 저장 형식 | 미정 (JSONL/CSV/SQLite 등 추후 결정) |

## 모델 입력 스펙

| 항목 | 값 |
|------|----|
| 샘플링 | 100Hz |
| 윈도우 | 3초 = 300패킷 |
| 서브캐리어 수 (ESP32) | 52개 (LLTF) |
| Rx 결합 | Rx1(52) + Rx2(52)를 서브캐리어 축으로 concat → (300, 104) |
| 최종 모델 입력 shape | (N, 1, 28, 20) |

## 시스템 응답 시간 목표

| 구간 | 목표 |
|------|------|
| ESP32 → 추론 서버 | UDP 지연 최소화 |
| 전처리 + 추론 | 전체 ≤ 1.5초 |
| 사용자 체감 지연 | ≤ 3초 |
| SMS API (SOLAPI) | ≤ 1초 |
