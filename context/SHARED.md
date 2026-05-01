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
[device_id (1B)] [subcarrier_num (1B)] [reserved (2B)] [seq (4B)] [timestamp_us (8B)] [amplitude × 52 (208B)]
총 224B

| 필드 | 타입 | 크기 | 값 |
|------|------|------|----|
| device_id | uint8 | 1B | Rx1=0x01, Rx2=0x02 |
| subcarrier_num | uint8 | 1B | 52 |
| reserved | padding | 2B | 0x0000 |
| seq | uint32 | 4B | 부팅 시 0 초기화, 패킷마다 +1 |
| timestamp_us | uint64 | 8B | SNTP 기준 Unix time (μs) |
| amplitude | float32 × 52 | 208B | √(I²+Q²) |

> **주의:** timestamp_us는 SNTP 동기화된 Unix time 기준이어야 함.
> 현재 테스트에서 `time=1777521566s`로 찍히는 값은 2026년 Unix time(약 1746000000s)과 불일치.
> ESP 담당자에게 SNTP 동기화 실제 적용 여부 확인 필요.

## 추론 결과 포맷 (서버 → Pi4, WebSocket)

```json
{
  "label": "fall",          // "fall" | "normal" | "lying" | ...
  "confidence": 0.94,
  "timestamp_us": 1234567890
}
```

## 모델 입력 스펙

| 항목 | 값 |
|------|----|
| 샘플링 | 100Hz |
| 윈도우 | 3초 = 300패킷 |
| 서브캐리어 수 (ESP32) | 52개 (LLTF) |
| 최종 모델 입력 shape | (N, 1, 28, 20) |

## 시스템 응답 시간 목표

| 구간 | 목표 |
|------|------|
| ESP32 → 추론 서버 | UDP 지연 최소화 |
| 전처리 + 추론 | 전체 ≤ 1.5초 |
| 사용자 체감 지연 | ≤ 3초 |
| SMS API (SOLAPI) | ≤ 1초 |