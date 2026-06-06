# 항목4 더미 4-arm 평가 (within-subject in-domain augmentation 효과)

non-WALK paired test N=27 | nonfall test 180 | seeds [42, 43, 44, 45, 46]
> 경계선 N — 점추정 단정 금지, CI 중심. WALK/subtype exploratory. train+더미는 train-only(val/test 봉인).

## [selected_by_val] non-WALK paired test (5-seed mean±std)
arm | recall | FAR | F1 | fwd_recall | early_fire | timely_4s | late_tp
---|---|---|---|---|---|---|---
A_fixed_original | 0.933±0.015 | 0.499±0.053 | 0.356±0.018 | 0.052±0.055 | 0.000±0.000 | 0.933±0.015 | 0.000±0.000
B_onset_original | 0.919±0.043 | 0.360±0.058 | 0.428±0.029 | 0.022±0.018 | 0.000±0.000 | 0.919±0.043 | 0.000±0.000
C_fixed_augmented | 0.919±0.028 | 0.501±0.069 | 0.352±0.032 | 0.044±0.015 | 0.000±0.000 | 0.919±0.028 | 0.000±0.000
D_onset_augmented | 0.874±0.038 | 0.388±0.077 | 0.397±0.040 | 0.052±0.038 | 0.000±0.000 | 0.874±0.038 | 0.000±0.000

**B→D (onset: 더미 효과, 메인)** (Δ=aug−orig, N=27): McNemar p=0.109
- Δrecall: -0.044 [-0.119,+0.015] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δfar: +0.028 [-0.014,+0.071] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δtimely_4s: -0.044 [-0.119,+0.015] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δearly_fire_rate: +0.000 [+0.000,+0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δforward_recall: +0.030 [-0.022,+0.081] → 유의차 미검출(CI 0 포함, under-powered 가능)

**A→C (fixed: 더미 효과)** (Δ=aug−orig, N=27): McNemar p=0.688
- Δrecall: -0.014 [-0.052,+0.015] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δfar: +0.002 [-0.034,+0.037] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δtimely_4s: -0.014 [-0.052,+0.015] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δearly_fire_rate: +0.000 [+0.000,+0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δforward_recall: -0.007 [-0.044,+0.030] → 유의차 미검출(CI 0 포함, under-powered 가능)

**결합효과 (D−C)−(B−A)**: 더미가 onset alignment와 결합해 추가효과?
- Δrecall: -0.029 [-0.126,+0.044] → 유의차 미검출(CI 0 포함, under-powered 가능)
- ΔFAR: +0.026 [-0.019,+0.072] → 유의차 미검출(CI 0 포함, under-powered 가능)

## [fixed_baseline] non-WALK paired test (5-seed mean±std)
arm | recall | FAR | F1 | fwd_recall | early_fire | timely_4s | late_tp
---|---|---|---|---|---|---|---
A_fixed_original | 0.933±0.015 | 0.499±0.053 | 0.356±0.018 | 0.052±0.055 | 0.000±0.000 | 0.933±0.015 | 0.000±0.000
B_onset_original | 0.919±0.043 | 0.352±0.028 | 0.431±0.010 | 0.022±0.018 | 0.000±0.000 | 0.919±0.043 | 0.000±0.000
C_fixed_augmented | 0.919±0.028 | 0.472±0.064 | 0.365±0.033 | 0.044±0.015 | 0.000±0.000 | 0.919±0.028 | 0.000±0.000
D_onset_augmented | 0.867±0.038 | 0.361±0.058 | 0.408±0.030 | 0.037±0.033 | 0.000±0.000 | 0.867±0.038 | 0.000±0.000

**B→D (onset: 더미 효과, 메인)** (Δ=aug−orig, N=27): McNemar p=0.065
- Δrecall: -0.051 [-0.133,+0.015] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δfar: +0.009 [-0.032,+0.050] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δtimely_4s: -0.051 [-0.133,+0.015] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δearly_fire_rate: +0.000 [+0.000,+0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δforward_recall: +0.015 [-0.030,+0.059] → 유의차 미검출(CI 0 포함, under-powered 가능)

**A→C (fixed: 더미 효과)** (Δ=aug−orig, N=27): McNemar p=0.688
- Δrecall: -0.014 [-0.052,+0.015] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δfar: -0.027 [-0.063,+0.008] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δtimely_4s: -0.014 [-0.052,+0.015] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δearly_fire_rate: +0.000 [+0.000,+0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δforward_recall: -0.007 [-0.044,+0.030] → 유의차 미검출(CI 0 포함, under-powered 가능)

**결합효과 (D−C)−(B−A)**: 더미가 onset alignment와 결합해 추가효과?
- Δrecall: -0.036 [-0.133,+0.037] → 유의차 미검출(CI 0 포함, under-powered 가능)
- ΔFAR: +0.036 [-0.007,+0.081] → 유의차 미검출(CI 0 포함, under-powered 가능)

## class ratio / effective sampled mass (train)
arm | train fall | fall:nonfall | fall effective mass(source0.60)
---|---|---|---
A_fixed_original | 223 | 0.1517 | 0.0694
B_onset_original | 114 | 0.0776 | 0.0376
C_fixed_augmented | 524 | 0.3565 | 0.1411
D_onset_augmented | 380 | 0.2585 | 0.1094

## 결론 범위
- within-subject / in-domain augmentation 효과로 제한. 새 subject/env 일반화 해석 금지.
- 더미는 train non-WALK fall origin 기반, train-only. val/test 원본 봉인.
