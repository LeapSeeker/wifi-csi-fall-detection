# manifest_v2_manual_augmented 요약

총 360 세션 | v1 360 + 검수 115(high 98 + normal 17) 반영.
low_quality 판정: per-session noise>1.206 OR mad>0.079 (환경 일괄 분류 금지).

## onset_status 분포
- auto_reviewed: 214
- manual_corrected: 50
- onset_unusable: 65
- pending_manual: 20
- data_invalid: 11

## usable_for_fixed = 349 / 360
split: train 223, val 55, test 71

## usable_for_onset_aligned = 264 / 360  ← 항목4 대상
split: train 168, val 45, test 51

### subtype별 usable_for_onset_aligned (해석 tier)
subtype | n | tier
---|---|---
FALL_WALK_F | 39 | 제한적해석(20+)
FALL_WALK_B | 37 | 제한적해석(20+)
FALL_SIT_F | 46 | 제한적해석(20+)
FALL_SIT_B | 40 | 제한적해석(20+)
FALL_STD_F | 51 | 제한적해석(20+)
FALL_STD_B | 51 | 제한적해석(20+)

### env×subject별 usable_for_onset_aligned
- E1_S01: 51
- E2_S02: 38
- E3_S03: 35
- E4_S01: 43
- E4_S02: 47
- E4_S03: 50

### ★ non-WALK pooled (train+val) usable_for_onset_aligned = 148 → 제한적해석(20+)
   WALK pooled (train+val) = 65 → 제한적해석(20+)

## exclude_reason_primary 분포 (제외 76)
- walking_residual: 6
- beep_misfire: 7
- no_clear_transient: 43
- low_quality_env_subject: 9
- data_short_or_corrupt: 11

### low_quality_env_subject 세션 (9) — 세션단위 판정
- E2_S02 FALL_STD_B noise=1.15083 E2_S02_A_FALL_STD_B_T002.csv
- E2_S02 FALL_WALK_F noise=1.26741 E2_S02_A_FALL_WALK_F_T007.csv
- E2_S02 FALL_WALK_F noise=1.55929 E2_S02_A_FALL_WALK_F_T009.csv
- E2_S02 FALL_WALK_F noise=1.20737 E2_S02_A_FALL_WALK_F_T010.csv
- E4_S02 FALL_WALK_B noise=1.2016 E4_S02_A_FALL_WALK_B_T003.csv
- E4_S02 FALL_WALK_B noise=1.17836 E4_S02_A_FALL_WALK_B_T005.csv
- E4_S03 FALL_SIT_B noise=1.20565 E4_S03_A_FALL_SIT_B_T005.csv
- E4_S03 FALL_STD_B noise=1.30693 E4_S03_A_FALL_STD_B_T001.csv
- E4_S03 FALL_WALK_F noise=1.18674 E4_S03_A_FALL_WALK_F_T010.csv

## final onset median (clean, imputation 없음)
- overall: 134.0  (manual 150.0 / auto 132.5)
- by_subtype: {'FALL_WALK_F': 136.5, 'FALL_WALK_B': 120.0, 'FALL_SIT_F': 132.5, 'FALL_SIT_B': 130.5, 'FALL_STD_F': 142.0, 'FALL_STD_B': 144.0}
- by_split: {'train': 130.0, 'val': 143.0, 'test': 134.0, 'out_of_scope': None}
- ⚠ median imputation 금지: 제외 세션에 median onset 미부여(편향 유입 방지).

## 항목4 진행 가능성 소견
- non-WALK pooled(train+val) 148개 → **제한적해석(20+)**. 메인 paired 비교 가능.
- WALK 은 65개 → 제한적해석(20+). subtype별로는 더 적어 별도 해석 주의.
- fixed 는 onset_unusable 포함 usable_for_fixed=true 라 데이터 손실 없음.
