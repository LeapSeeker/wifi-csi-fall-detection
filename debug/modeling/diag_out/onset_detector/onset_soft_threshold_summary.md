# 게이트 2 soft threshold 분위수 비교 (nominal k3/s5 고정, train+val, test 봉인)

train+val 세션: 287  | review budget: 35.0%

## 분위수 세트별 review
set | bottom/top | hard% | total_review% | soft증분 | WALK_B% | WALK_F% | SIT_F%
A_10_90 | 10/90 | 33.4 | 43.6 | +29 | 67.3 | 57.4 | 31.7
B_15_85 | 15/85 | 33.4 | 50.2 | +48 | 73.1 | 61.7 | 43.9
C_20_80 | 20/80 | 33.4 | 59.2 | +74 | 84.6 | 72.3 | 53.7

**권장 세트: A_10_90** (total_review ≤ 35.0% 중 가장 느슨, 모두 초과 시 10/90)

## soft_warning_count 분포 (세트별)
- A_10_90: {0: 151, 1: 84, 2: 41, 3: 8, 4: 3}
- B_15_85: {0: 114, 1: 94, 2: 52, 3: 23, 4: 4}
- C_20_80: {0: 85, 1: 88, 2: 80, 3: 23, 4: 11}

## subtype별 review% (권장 세트 A_10_90)
- FALL_SIT_B: review 44.0% (hard 34.0%, n=50)
- FALL_SIT_F: review 31.7% (hard 22.0%, n=41)
- FALL_STD_B: review 24.5% (hard 12.2%, n=49)
- FALL_STD_F: review 33.3% (hard 31.2%, n=48)
- FALL_WALK_B: review 67.3% (hard 55.8%, n=52)
- FALL_WALK_F: review 57.4% (hard 42.6%, n=47)

## 대표 plot 카테고리 (plots_soft/)
- broad_only: ['E1_S01_A_FALL_SIT_B_T005.csv', 'E1_S01_A_FALL_SIT_B_T008.csv']
- walkb_early: ['E1_S01_A_FALL_WALK_B_T001.csv', 'E2_S02_A_FALL_WALK_B_T008.csv', 'E4_S01_A_FALL_WALK_B_T007.csv']
- rise_not_found: ['E1_S01_A_FALL_SIT_B_T005.csv', 'E1_S01_A_FALL_SIT_B_T008.csv', 'E1_S01_A_FALL_SIT_B_T009.csv']
- high_noise: ['E4_S02_A_FALL_WALK_B_T007.csv', 'E1_S01_A_FALL_WALK_F_T002.csv', 'E1_S01_A_FALL_WALK_F_T004.csv']
