# firmware/CLAUDE.md — ESP32-S3 펌웨어
# 담당: 주화 | 브랜치: feature/esp32-firmware, feature/data-collection
# 공유 인터페이스 → context/SHARED.md

## 하드웨어 구성

| 장치 | 역할 |
|------|------|
| ESP32-S3 DEVKITC-1U-N8R8 (Tx) | 비콘 송신 |
| ESP32-S3 DEVKITC-1U-N8R8 (Rx1, Rx2) | CSI 수신 + UDP 전송 |

- 안테나: 2.4GHz 외부 헬리컬 안테나
- 전원: 포터블 배터리 (3대 모두)
- 개발 환경: ESP-IDF (C)

## CSI 설정

| 항목 | 값 |
|------|----|
| 활성화 서브타입 | LLTF only |
| 서브캐리어 수 | 52개 |
| 샘플링 주파수 | 100Hz (비콘 인터벌 10ms) |
| 진폭 추출 위치 | Rx 단에서 √(I²+Q²) 계산 |
| 전송 데이터 | float 진폭 (raw IQ 전송 아님) |

## UDP 전송 (Rx → 추론 서버)

패킷 구조 및 포트 → `context/SHARED.md` 참조

```c
// 전송 예시 구조체
typedef struct {
    uint8_t  device_id;       // Rx1=0x01, Rx2=0x02
    uint32_t seq_num;
    uint64_t timestamp_us;    // SNTP 기준
    uint16_t num_subcarriers; // 52
    float    amplitude[52];
} csi_packet_t;
```

## 시간 동기화

- SNTP 서버로 Rx1/Rx2 타임스탬프 동기화 필수
- 동기화 주기: 부팅 시 1회 + 주기적 재동기화 권장

## 참고 자료

- ESP32-CSI-Tool: https://github.com/StevenMHernandez/ESP32-CSI-Tool
- ESP-IDF 공식 문서: https://docs.espressif.com/projects/esp-idf/en/latest/

## TODO

- [ ] 포터블 라우터 보유 여부 확인 후 네트워크 구성 확정
- [ ] 배터리 실측 운용 시간 수치화