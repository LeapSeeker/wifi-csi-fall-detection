# 항목4 event-level 평가 결과 (fixed vs onset_primary alignment)

단위 N: {'nonwalk_paired': 27, 'walk_paired': 8, 'all_test_fall': 71} | non-fall test 180 | seeds [42, 43, 44, 45, 46]
정의: timely 0<=lat<=3/4, early<0, late>4; latency time_origin=onset_orig

> 경계선 N(non-WALK paired test ~27) — 점추정 단정 금지, CI 중심. WALK/subtype exploratory.

## [selected_by_val] non-WALK paired (main)
policy | recall | FAR | F1 | fwd_recall | early_fire | timely_4s | late_tp
---|---|---|---|---|---|---|---
fixed | 0.948±0.018 | 0.509±0.048 | 0.357±0.026 | 0.037±0.023 | 0.000±0.000 | 0.948±0.018 | 0.000±0.000
onset_primary | 0.919±0.036 | 0.338±0.023 | 0.441±0.017 | 0.022±0.018 | 0.000±0.000 | 0.919±0.036 | 0.000±0.000
onset_reduced | 0.889±0.070 | 0.409±0.049 | 0.386±0.022 | 0.104±0.036 | 0.000±0.000 | 0.889±0.070 | 0.000±0.000

**통계 (onset_primary − fixed, non-WALK paired N=27)**
- McNemar: fixed-only fire 6, onset-only 2, p=0.289
- Δrecall: -0.029 [95%CI -0.074, +0.007] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δearly_fire_rate: +0.000 [95%CI +0.000, +0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δtimely_4s: -0.029 [95%CI -0.074, +0.007] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δfar: -0.172 [95%CI -0.218, -0.126] → 유의(CI 0 미포함)
- Δforward_recall: -0.015 [95%CI -0.052, +0.015] → 유의차 미검출(CI 0 포함, under-powered 가능)

## [fixed_baseline] non-WALK paired (main)
policy | recall | FAR | F1 | fwd_recall | early_fire | timely_4s | late_tp
---|---|---|---|---|---|---|---
fixed | 0.948±0.018 | 0.509±0.048 | 0.357±0.026 | 0.037±0.023 | 0.000±0.000 | 0.948±0.018 | 0.000±0.000
onset_primary | 0.874±0.038 | 0.288±0.029 | 0.462±0.023 | 0.000±0.000 | 0.000±0.000 | 0.874±0.038 | 0.000±0.000
onset_reduced | 0.844±0.079 | 0.339±0.062 | 0.413±0.013 | 0.089±0.030 | 0.000±0.000 | 0.844±0.079 | 0.000±0.000

**통계 (onset_primary − fixed, non-WALK paired N=27)**
- McNemar: fixed-only fire 10, onset-only 0, p=0.002
- Δrecall: -0.073 [95%CI -0.133, -0.022] → 유의(CI 0 미포함)
- Δearly_fire_rate: +0.000 [95%CI +0.000, +0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)
- Δtimely_4s: -0.073 [95%CI -0.133, -0.022] → 유의(CI 0 미포함)
- Δfar: -0.222 [95%CI -0.272, -0.173] → 유의(CI 0 미포함)
- Δforward_recall: -0.037 [95%CI -0.089, +0.000] → 유의차 미검출(CI 0 포함, under-powered 가능)

## WALK exploratory — early_fire_rate (정렬이 전조 오발화 줄이는지)
- [selected_by_val] fixed: early_fire_rate 0.000±0.000
- [selected_by_val] onset_primary: early_fire_rate 0.000±0.000
- [selected_by_val] onset_reduced: early_fire_rate 0.000±0.000

## Gate3 발견 반영
- onset_primary OOB 88 > onset_reduced 30 (늦은 onset, onset+250>clean400 끝)
- usable_for_onset_aligned 264 중 clean onset 263 (1건 beep구간 수동 onset null)
