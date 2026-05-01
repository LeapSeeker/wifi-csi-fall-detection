# model/CLAUDE.md — AI 모델 / 전처리 / 데이터 수집
# 담당: 진규 | 브랜치: feature/pretrained-model, feature/augmentation, feature/finetune
# 공유 인터페이스 → context/SHARED.md

## 전처리 파이프라인: RPCA → ACF → SDP

| 단계 | 파라미터 |
|------|----------|
| RPCA | λ = 1/√max(N_T, N_S), S(sparse) 성분만 사용 |
| ACF | N_Δ = 20 |
| SDP | 서브윈도우 W=30, stride=10, W_T=28 |
| 모델 입력 | (N, 1, 28, 20) |

서브캐리어 수 불일치(UT-HAR 90개 vs ESP32 52개)는 SDP 집계 단계에서 해소됨.

## 모델: CNN+GRU+Attention

- Temporal Attention 포함 필수 (CNN-LSTM, CNN-GRU 단독 아님)
- 참고: XFall (IEEE JSAC 2024) — Attention + SDP 조합 검증

## 사전학습 전략 (전략 B)

UT-HAR raw 데이터에 동일 파이프라인 직접 적용 후 학습.
UT-HAR (1000Hz)
→ Butterworth 4차 LPF (45Hz 컷오프, filtfilt)
→ 1/10 데시메이션 → 100Hz
→ RPCA → ACF → SDP
→ CNN+GRU+Attention 사전학습
- UT-HAR raw: https://github.com/ermongroup/Wifi_Activity_Recognition
- SenseFi processed 버전 미사용 (파이프라인 불일치)
- 가중치 저장: `model/pretrained/` (100MB 초과 시 Git LFS)

## 파인튜닝

- 데이터: 자체 수집 ESP32 데이터 (210세션)
- 결과 저장: `model/finetune/`
- ONNX 변환 후 `model/export/`에 저장

## 데이터 수집 전략

| 항목 | 값 |
|------|----|
| 목표 | 7클래스 × 30세션 = 210세션 |
| 증강 | 5배 → 약 1,050샘플 |
| 증강 기법 | Gaussian noise, time shift, amplitude scaling |

클래스: 낙상(앞/뒤/옆 각 10회), 걷기, 앉기/일어서기, 눕기, 제자리 서기, 달리기(빠른 보행), 물건 집기

`data/collection_log.md` 기록 양식:

```markdown
| 날짜 | 피험자 | 동작 | 횟수 | 환경 | 파일명 |
```

## 참고 자료

- SenseFi benchmark: https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
- robust-pca: https://github.com/dganguli/robust-pca
- XFall (IEEE JSAC 2024): https://ieeexplore.ieee.org/document/10438965

## TODO

- [ ] 슬라이딩 윈도우 크기 실측 후 확정