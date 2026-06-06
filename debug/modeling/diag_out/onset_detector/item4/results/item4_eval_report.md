# 항목4 event-level 평가 결과 (fixed vs onset_primary alignment)

단위 N: {'nonwalk_paired': 27, 'walk_paired': 8, 'all_test_fall': 71} | non-fall test 180 | seeds [42, 43, 44, 45, 46]
정의: timely 0<=lat<=3/4, early<0, late>4; latency time_origin=onset_orig

> 경계선 N(non-WALK paired test ~27) — 점추정 단정 금지, CI 중심. WALK/subtype exploratory.

## [selected_by_val] non-WALK paired (main)
policy | recall | FAR | F1 | fwd_recall | early_fire | timely_4s | late_tp
---|---|---|---|---|---|---|---
fixed | 0.933±0.015 | 0.499±0.053 | 0.356±0.018 | 0.052±0.055 | 0.000±0.000 | 0.933±0.015 | 0.000±0.000
onset_primary | 0.919±0.043 | 0.360±0.058 | 0.428±0.029 | 0.022±0.018 | 0.000±0.000 | 0.919±0.043 | 0.000±0.000
onset_reduced | 0.896±0.092 | 0.416±0.045 | 0.384±0.019 | 0.170±0.109 | 0.000±0.000 | 0.896±0.092 | 0.000±0.000

**통계 (onset_primary − fixed, non-WALK paired N=27)**
- McNemar: fixed-only fire 5, onset-only 3, p=0.727
- Δrecall: -0.015 [95%CI -0.059, +0.030] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δearly_fire_rate: +0.000 [95%CI +0.000, +0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δtimely_4s: -0.015 [95%CI -0.059, +0.030] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δfar: -0.139 [95%CI -0.191, -0.091] → 유의(CI 0 미포함)
- Δforward_recall: -0.029 [95%CI -0.081, +0.022] → 유의차 미검출(CI 0 포함, under-powered 가능)

## [fixed_baseline] non-WALK paired (main)
policy | recall | FAR | F1 | fwd_recall | early_fire | timely_4s | late_tp
---|---|---|---|---|---|---|---
fixed | 0.933±0.015 | 0.499±0.053 | 0.356±0.018 | 0.052±0.055 | 0.000±0.000 | 0.933±0.015 | 0.000±0.000
onset_primary | 0.919±0.043 | 0.352±0.028 | 0.431±0.010 | 0.022±0.018 | 0.000±0.000 | 0.919±0.043 | 0.000±0.000
onset_reduced | 0.881±0.079 | 0.379±0.049 | 0.401±0.036 | 0.141±0.059 | 0.000±0.000 | 0.881±0.079 | 0.000±0.000

**통계 (onset_primary − fixed, non-WALK paired N=27)**
- McNemar: fixed-only fire 5, onset-only 3, p=0.727
- Δrecall: -0.015 [95%CI -0.059, +0.030] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δearly_fire_rate: +0.000 [95%CI +0.000, +0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δtimely_4s: -0.015 [95%CI -0.059, +0.030] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δfar: -0.147 [95%CI -0.198, -0.100] → 유의(CI 0 미포함)
- Δforward_recall: -0.029 [95%CI -0.081, +0.022] → 유의차 미검출(CI 0 포함, under-powered 가능)

## WALK exploratory — early_fire_rate (정렬이 전조 오발화 줄이는지)
- [selected_by_val] fixed: early_fire_rate 0.000±0.000
- [selected_by_val] onset_primary: early_fire_rate 0.000±0.000
- [selected_by_val] onset_reduced: early_fire_rate 0.000±0.000

## ★ val recall–FAR frontier (배포 운영점별 달성 가능 최대 recall, 5-seed mean)
policy | FAR≤0.15 | FAR≤0.20 | FAR≤0.30 | max-F1
---|---|---|---|---
fixed | R0.665/FAR0.131 | R0.785/FAR0.195 | R0.851/FAR0.297 | F10.679(R0.785/FAR0.195)
onset_primary | R0.767/FAR0.147 | R0.829/FAR0.192 | R0.931/FAR0.284 | F10.714(R0.887/FAR0.219)
onset_reduced | R0.698/FAR0.145 | R0.800/FAR0.188 | R0.858/FAR0.252 | F10.694(R0.800/FAR0.188)

> 배포 운영점은 frontier에서 선택. D-023 selected 표는 목표(FAR≤0.15) 미달 시 max-recall fallback이라 FAR 높음.

## Gate3 발견 반영
- onset_primary OOB 88 > onset_reduced 30 (늦은 onset, onset+250>clean400 끝)
- usable_for_onset_aligned 264 중 clean onset 263 (1건 beep구간 수동 onset null)
