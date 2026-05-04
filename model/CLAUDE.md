# model/CLAUDE.md — AI 모델 / 전처리 / 데이터 수집
# 담당: 진규 | 브랜치: feature/pretrained-model, feature/augmentation, feature/finetune
# 공유 인터페이스 → context/SHARED.md

## 전처리 파이프라인: RPCA → ACF → SDP

| 단계 | 파라미터 |
|------|----------|
| RPCA | λ = 1/√max(N_T, N_S), S(sparse) 성분만 사용 |
| ACF  | N_Δ = 20, ΔT = 0.01s |
| SDP  | 서브윈도우 W=30, stride=10, W_T=28 |
| 모델 입력 | (N, 1, 28, 20) |

서브캐리어 수 불일치(Alsaify 90개 vs ESP32 52개)는 SDP 집계 단계에서 해소됨.

## 모델: CNN+GRU+Attention

- Temporal Attention 포함 필수 (CNN-LSTM, CNN-GRU 단독 아님)
- 참고: XFall (IEEE JSAC 2024) — Attention + SDP 조합 검증

## 사전학습 전략

사전학습 데이터셋: **Alsaify LOS 환경 (E1+E2, 피험자 20명 × 20회)**
- NLOS 환경(E3) 제외 — 배포 환경(LOS)과 도메인 불일치. 추후 확장 가능성으로만 보존.

Alsaify raw CSV (복소수 I+jQ)
→ 진폭 추출: sqrt(I²+Q²)
→ 다운샘플링: 320Hz → 100Hz (resample_poly(up=5, down=16))
→ 슬라이딩 윈도우: 300패킷 단위
→ RPCA → ACF → SDP
→ CNN+GRU+Attention 사전학습

```python
from scipy.signal import resample_poly

def downsample_alsaify(data):
    # data shape: (m패킷, 90)
    return resample_poly(data, up=5, down=16, axis=0)
    # 출력: (400패킷, 90)
```

**SafeSignal 클래스 매핑 (사전학습 기준)**

| SafeSignal 클래스 | Alsaify 활동 코드 | 샘플 수 (E1+E2) |
|---|---|---|
| fall | A2, A5 | 400 + 400 = 800 |
| walking | A6, A8 | 400 + 400 = 800 |
| sit_stand | A10, A11 | 400 + 400 = 800 |
| lying | A3 (C1+C2 통합) | 400 + 400 = 800 |
| standing | A4 (C2+C4 통합) | 400 + 400 = 800 |
| picking | A12 | 400 |
| **합계** | | **4,400** |

달리기(빠른 보행): Alsaify 대응 없음 → 파인튜닝 전용

가중치 저장: `model/pretrained/` (100MB 초과 시 Git LFS)

## 파인튜닝

- 데이터: 자체 수집 ESP32 데이터 (240세션)
- 결과 저장: `model/finetune/`
- ONNX 변환 후 `model/export/`에 저장

## 데이터 수집 전략

| 항목 | 값 |
|------|----|
| 목표 | 240세션 |
| 증강 | 5배 → 약 1,200샘플 |
| 증강 기법 | Jittering / Scaling / Time Warping / Noise+Scale 조합 |

**클래스 구성 (낙상은 수집 구조만 세분화, 학습 레이블은 단일 클래스)**

| 동작 클래스 | 수집 구조 | 횟수 |
|---|---|---|
| 낙상 — 앉다가 | 앞/뒤/옆 각 10회 | 30회 |
| 낙상 — 서다가 | 앞/뒤/옆 각 10회 | 30회 |
| 걷기 | — | 30회 |
| 앉기/일어서기 | — | 30회 |
| 눕기 | — | 30회 |
| 제자리 서기 | — | 30회 |
| 달리기(빠른 보행) | — | 30회 |
| 물건 집기 | — | 30회 |
| **합계** | | **240세션** |

`data/collection_log.md` 기록 양식:
```markdown
| 날짜 | 피험자 | 동작 | 방향 | 횟수 | 환경 | 파일명 |
```

## 다운샘플링 근거

- ESP32-S3 샘플링 레이트: 100Hz (비콘 간격 10ms)
- 인체 동작 최대 도플러 편이: 32Hz (2.4GHz, 이동속도 2m/s 상한 기준)
- 나이퀴스트 한계(50Hz) > 32Hz → 100Hz로 충분
- 참고: https://arxiv.org/pdf/1611.01801

## 참고 자료

- Alsaify dataset (Mendeley): https://data.mendeley.com/datasets/v38wjmz6f6/1
- Alsaify 논문 (PMC): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7704290/
- Alsaify GitHub: https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments
- robust-pca: https://github.com/dganguli/robust-pca
- XFall (IEEE JSAC 2024): https://ieeexplore.ieee.org/document/10438965

## TODO

- [ ] 슬라이딩 윈도우 크기 실측 후 확정
- [ ] 분산 Rx 안테나(ESP32-B/C) 신호 통합 방식 결정 (2채널 vs 평균)
- [ ] 자체수집 계획 개편안 확정 (W3 이전)