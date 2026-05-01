# SafeSignal — CLAUDE.md

WiFi CSI 기반 낙상 감지 시스템. 카메라 없이 WiFi 신호 변화만으로 낙상을 감지하며,
1인 거주 노인 가구 대상. 낙상 감지 시 보호자에게 SMS 알림 전송.

- 팀명: MATE | 데모: 2026-06-04 | 최종 발표: 2026-06-11
- 추론 목표: 낙상 재현율(MVG) ≥ 85%, FAR ≤ 15%, F1 ≥ 0.85

## 저장소 구조
wifi-csi-fall-detection/
├── CLAUDE.md                  ← 현재 파일 (진입점)
├── context/
│   └── SHARED.md              ← 파트 공통 인터페이스 정의
├── firmware/
│   ├── CLAUDE.md              ← ESP32-S3 펌웨어 작업 시 참조
│   ├── tx/
│   └── rx/
├── model/
│   ├── CLAUDE.md              ← AI 모델 / 전처리 / 데이터 작업 시 참조
│   ├── pretrained/
│   ├── finetune/
│   ├── augment/
│   └── export/
├── rpi4/
│   └── CLAUDE.md              ← Raspberry Pi 4 파이프라인 작업 시 참조
├── data/
│   ├── raw/
│   ├── augmented/
│   └── collection_log.md
└── docs/

## 작업 진입점

|                 작업 내용               |     읽어야 할 파일    |
|-----------------------------------------|----------------------|
| ESP32 펌웨어, CSI 수집, UDP 송신         | `firmware/CLAUDE.md` |
| 전처리 파이프라인, 모델 학습, 데이터 수집 | `model/CLAUDE.md`    |
| Pi4 수신, 오디오 출력, 버튼, SMS 알림    | `rpi4/CLAUDE.md`     |
| 포트/패킷 구조/IP 등 파트 간 공유 정보   | `context/SHARED.md`   |

## 브랜치 구조
main
├── feature/esp32-firmware      ← 주화
├── feature/data-collection     ← 주화
├── feature/pretrained-model    ← 진규
├── feature/augmentation        ← 진규
├── feature/finetune            ← 진규
└── feature/rpi4-pipeline       ← 동석
커밋 규칙: `기능 설명 추가` / `[수정] 내용` / `[데이터] 내용` / `[결과] 내용`