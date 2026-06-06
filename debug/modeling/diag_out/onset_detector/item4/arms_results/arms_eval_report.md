# 항목4 더미 4-arm 평가 (within-subject in-domain augmentation 효과)

non-WALK paired test N=27 | nonfall test 180 | seeds [42, 43, 44, 45, 46]
> 경계선 N — 점추정 단정 금지, CI 중심. WALK/subtype exploratory. train+더미는 train-only(val/test 봉인).

## [selected_by_val] non-WALK paired test (5-seed mean±std)
arm | recall | FAR | F1 | fwd_recall | early_fire | timely_4s | late_tp
---|---|---|---|---|---|---|---
A_fixed_original | 0.933±0.015 | 0.499±0.053 | 0.356±0.018 | 0.052±0.055 | 0.000±0.000 | 0.933±0.015 | 0.000±0.000
B_onset_original | 0.919±0.043 | 0.360±0.058 | 0.428±0.029 | 0.022±0.018 | 0.000±0.000 | 0.919±0.043 | 0.000±0.000
C_fixed_augmented | 0.933±0.015 | 0.491±0.033 | 0.359±0.017 | 0.015±0.018 | 0.000±0.000 | 0.933±0.015 | 0.000±0.000
D_onset_augmented | 0.926±0.023 | 0.470±0.051 | 0.367±0.020 | 0.015±0.030 | 0.000±0.000 | 0.926±0.023 | 0.000±0.000

**B→D (onset: 더미 효과, 메인)** (Δ=aug−orig, N=27): McNemar p=1.000
- Δrecall: +0.008 [-0.022,+0.037] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δfar: +0.110 [+0.070,+0.151] → 유의(CI 0 미포함)
- Δtimely_4s: +0.008 [-0.022,+0.037] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δearly_fire_rate: +0.000 [+0.000,+0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δforward_recall: -0.007 [-0.037,+0.015] → 유의차 미검출(CI 0 포함, under-powered 가능)

**A→C (fixed: 더미 효과)** (Δ=aug−orig, N=27): McNemar p=1.000
- Δrecall: +0.000 [-0.022,+0.022] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δfar: -0.007 [-0.042,+0.028] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δtimely_4s: +0.000 [-0.022,+0.022] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δearly_fire_rate: +0.000 [+0.000,+0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δforward_recall: -0.037 [-0.074,-0.007] → 유의(CI 0 미포함)

**결합효과 (D−C)−(B−A)**: 더미가 onset alignment와 결합해 추가효과?
- Δrecall: +0.007 [-0.022,+0.044] → 유의차 미검출(CI 0 포함, under-powered 가능)
- ΔFAR: +0.118 [+0.073,+0.166] → 유의(CI 0 미포함)

## [fixed_baseline] non-WALK paired test (5-seed mean±std)
arm | recall | FAR | F1 | fwd_recall | early_fire | timely_4s | late_tp
---|---|---|---|---|---|---|---
A_fixed_original | 0.933±0.015 | 0.499±0.053 | 0.356±0.018 | 0.052±0.055 | 0.000±0.000 | 0.933±0.015 | 0.000±0.000
B_onset_original | 0.919±0.043 | 0.352±0.028 | 0.431±0.010 | 0.022±0.018 | 0.000±0.000 | 0.919±0.043 | 0.000±0.000
C_fixed_augmented | 0.926±0.000 | 0.499±0.045 | 0.354±0.020 | 0.015±0.018 | 0.000±0.000 | 0.926±0.000 | 0.000±0.000
D_onset_augmented | 0.919±0.028 | 0.470±0.051 | 0.365±0.017 | 0.015±0.030 | 0.000±0.000 | 0.919±0.028 | 0.000±0.000

**B→D (onset: 더미 효과, 메인)** (Δ=aug−orig, N=27): McNemar p=1.000
- Δrecall: +0.001 [-0.030,+0.030] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δfar: +0.118 [+0.079,+0.159] → 유의(CI 0 미포함)
- Δtimely_4s: +0.001 [-0.030,+0.030] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δearly_fire_rate: +0.000 [+0.000,+0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δforward_recall: -0.007 [-0.037,+0.015] → 유의차 미검출(CI 0 포함, under-powered 가능)

**A→C (fixed: 더미 효과)** (Δ=aug−orig, N=27): McNemar p=1.000
- Δrecall: -0.007 [-0.022,+0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δfar: +0.000 [-0.036,+0.036] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δtimely_4s: -0.007 [-0.022,+0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δearly_fire_rate: +0.000 [+0.000,+0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δforward_recall: -0.037 [-0.074,-0.007] → 유의(CI 0 미포함)

**결합효과 (D−C)−(B−A)**: 더미가 onset alignment와 결합해 추가효과?
- Δrecall: +0.007 [-0.022,+0.044] → 유의차 미검출(CI 0 포함, under-powered 가능)
- ΔFAR: +0.118 [+0.073,+0.164] → 유의(CI 0 미포함)

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
