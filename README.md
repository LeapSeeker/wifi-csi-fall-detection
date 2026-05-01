# SafeSignal — WiFi CSI 기반 낙상 감지 시스템

카메라 없이 WiFi 신호 변화만으로 낙상을 감지하는 시스템.
1인 거주 노인 가구를 대상으로 하며, 낙상 감지 시 보호자에게 SMS 알림을 전송한다.

- 팀명: MATE
- 데모: 2026-06-04 / 최종 발표: 2026-06-11

---

## 시스템 구성

                    공간(사람/장애물 통과)
                   ┌────────────────────┐
ESP32-S3 Tx ──공중 송출──────────────────► ESP32-S3 Rx1 ─┐
(WiFi 프레임 브로드캐스트)                                ├─ UDP(5005) ──► 추론 서버
                   └────────────────────► ESP32-S3 Rx2 ─┘                   │
                                                                        WebSocket
                        [WiFi 공유기]                                       │
                     (네트워크 인프라만)                               Raspberry Pi 4
                                                               (오디오 출력 / 버튼 / SMS)


## 저장소 구조 및 작업 가이드

작업 시작 전 해당 디렉토리의 `CLAUDE.md`를 반드시 먼저 읽을 것.
wifi-csi-fall-detection/
├── CLAUDE.md
├── context/
│   └── SHARED.md
├── firmware/
│   ├── CLAUDE.md
│   ├── tx/
│   └── rx/
├── model/
│   ├── CLAUDE.md
│   ├── pretrained/
│   ├── finetune/
│   ├── augment/
│   └── export/
├── rpi4/
│   └── CLAUDE.md
└── data/
├── raw/
├── augmented/
└── collection_log.md

---

### `context/`

파트 간 공유 인터페이스 정의. **수정 시 반드시 팀 전체에 공유.**

| 파일 | 내용 |
|------|------|
| `SHARED.md` | UDP 패킷 구조, 포트 번호, IP 설정, 모델 입력 스펙, 추론 결과 포맷 |

---

### `firmware/`

담당: 주화 | 브랜치: `feature/esp32-firmware`, `feature/data-collection`

**`firmware/tx/`**

| 파일 | 작성 내용 |
|------|-----------|
| `main.c` | ESP32-S3 Tx 메인 엔트리. WiFi 초기화, 비콘 송신 설정 |
| `wifi_config.h` | SSID, 비밀번호, 채널, 비콘 인터벌(10ms) 설정 |

**`firmware/rx/`**

| 파일 | 작성 내용 |
|------|-----------|
| `main.c` | ESP32-S3 Rx 메인 엔트리. CSI 콜백 등록, UDP 전송 루프 |
| `csi_handler.c` | CSI 콜백 함수. LLTF 서브캐리어 52개 진폭 추출 (√(I²+Q²)) |
| `udp_sender.c` | CSI 패킷 구조체 직렬화 및 UDP 전송 |
| `sntp_sync.c` | SNTP 시간 동기화. 부팅 시 1회 + 주기적 재동기화 |
| `packet.h` | UDP 패킷 구조체 정의 (device_id, seq, timestamp_us, amplitude[52]) |

---

### `model/`

담당: 진규 | 브랜치: `feature/pretrained-model`, `feature/augmentation`, `feature/finetune`

**`model/pretrained/`**

| 파일 | 작성 내용 |
|------|-----------|
| `train_pretrain.py` | UT-HAR raw 데이터에 RPCA→ACF→SDP 적용 후 CNN+GRU+Attention 사전학습 |
| `preprocess_uthar.py` | UT-HAR 1000Hz → 100Hz 다운샘플링 (Butterworth LPF + 데시메이션) |
| `pipeline.py` | RPCA→ACF→SDP 전처리 파이프라인 공통 모듈 |
| `model.py` | CNN+GRU+Attention 아키텍처 정의 |
| `*.pth` | 사전학습 완료 가중치 (100MB 초과 시 Git LFS) |

**`model/augment/`**

| 파일 | 작성 내용 |
|------|-----------|
| `augment.py` | Gaussian noise, time shift, amplitude scaling 증강 구현. 210세션 → 1,050샘플 목표 |

**`model/finetune/`**

| 파일 | 작성 내용 |
|------|-----------|
| `train_finetune.py` | 자체 수집 ESP32 데이터로 파인튜닝. 사전학습 가중치 로드 후 fine-tuning |
| `dataset.py` | ESP32 수집 데이터 로더. 라벨 스키마, 윈도우 슬라이딩 처리 |
| `evaluate.py` | 재현율(MVG/Stretch), FAR, F1 평가 코드 |

**`model/export/`**

| 파일 | 작성 내용 |
|------|-----------|
| `export_onnx.py` | 파인튜닝 완료 모델 → ONNX 변환 스크립트 |
| `*.onnx` | 변환된 추론용 모델 (Git LFS) |

---

### `rpi4/`

담당: 동석(통신/알림), 주화(환경 설정/오디오) | 브랜치: `feature/rpi4-pipeline`

| 파일 | 작성 내용 |
|------|-----------|
| `main.py` | 메인 루프. WebSocket 연결 유지, 추론 결과 수신, 액션 디스패치 |
| `ws_client.py` | 추론 서버에 아웃바운드 WebSocket 연결. 재연결 로직 포함 |
| `audio_player.py` | 낙상 감지 시 사전 생성된 음성 파일 재생 (런타임 TTS 없음) |
| `button_handler.py` | 하드웨어 버튼 GPIO 처리. ①경보 취소 ②긴급 SMS 트리거 |
| `sms_alert.py` | SOLAPI SDK 연동. 자동/수동 SMS 발송 |
| `config.py` | 추론 서버 WebSocket URL, SOLAPI API 키, 수신자 번호 등 설정 |
| `audio/` | 사전 생성된 음성 알림 파일 (.wav, .mp3) |

---

### `data/`

담당: 진규 (수집 코드), 주화 (수집 실행)

| 파일/폴더 | 내용 |
|-----------|------|
| `raw/` | ESP32 수집 원시 CSI 데이터 (.csv). `.gitignore` 처리 — Git에 올리지 않음 |
| `augmented/` | 증강된 데이터 (.csv). `.gitignore` 처리 |
| `collection_log.md` | 수집 세션 기록. 날짜/피험자/동작/횟수/환경/파일명 형식으로 매 세션 기록 |

---

## 브랜치 전략
main
├── feature/esp32-firmware      ← 주화
├── feature/data-collection     ← 주화
├── feature/pretrained-model    ← 진규
├── feature/augmentation        ← 진규
├── feature/finetune            ← 진규
└── feature/rpi4-pipeline       ← 동석
각 브랜치 작업 완료 → PR 생성 → 팀장 리뷰 → `main` merge

## 커밋 메시지 규칙

| 유형 | 형식 | 예시 |
|------|------|------|
| 기능 추가 | `기능 설명 추가` | `Gaussian noise 증강 코드 추가` |
| 버그 수정 | `[수정] 문제 내용` | `[수정] UDP 패킷 파싱 오류 수정` |
| 데이터 | `[데이터] 내용` | `[데이터] P02 낙상 30회 수집 완료` |
| 실험 결과 | `[결과] 내용` | `[결과] 파인튜닝 val accuracy 0.91` |

## 참고 자료

- ESP32-CSI-Tool: https://github.com/StevenMHernandez/ESP32-CSI-Tool
- SenseFi benchmark: https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
- UT-HAR dataset: https://github.com/ermongroup/Wifi_Activity_Recognition
- robust-pca: https://github.com/dganguli/robust-pca
- XFall (IEEE JSAC 2024): https://ieeexplore.ieee.org/document/10438965
- SOLAPI 문서: https://docs.solapi.com