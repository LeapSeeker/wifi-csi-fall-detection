# Gate 3 event-centered cache 요약

세션 360 | clean400→window_to_model_input(동결) | window 300.
test split 은 생성하되 설계/threshold/crop 통계 결정 미사용.

## crop 생성 수 / crop_out_of_bounds
policy | generated | oob
---|---|---
fixed | 349 | 0
onset_primary | 175 | 88
onset_reduced | 233 | 30

onset_unusable 65 | data_invalid 11

## fixed by split / subtype
split: test 71, train 223, val 55
subtype: FALL_WALK_F 58, FALL_WALK_B 59, FALL_SIT_F 56, FALL_SIT_B 58, FALL_STD_F 58, FALL_STD_B 60

## onset_primary by split / subtype
split: test 35, train 114, val 26
subtype: FALL_WALK_F 21, FALL_WALK_B 26, FALL_SIT_F 33, FALL_SIT_B 31, FALL_STD_F 32, FALL_STD_B 32

## onset_reduced by split / subtype
split: test 48, train 146, val 39
subtype: FALL_WALK_F 33, FALL_WALK_B 30, FALL_SIT_F 42, FALL_SIT_B 37, FALL_STD_F 46, FALL_STD_B 45

## coverage (inclusion rate)
- fixed: WALK 117 / non-WALK 232
- onset_primary: WALK 47 / non-WALK 128

## ★ paired comparison 가능 N (fixed ∩ onset_primary)
- 전체 paired: 175
- **non-WALK pooled train+val: 101 → 제한적해석(20+)** (항목4 메인)
- WALK pooled train+val: 39 → 제한적해석(20+) (exploratory)

## env×subject 생성 수 (fixed)
- E1_S01: 60
- E2_S02: 56
- E3_S03: 57
- E4_S01: 59
- E4_S02: 58
- E4_S03: 59
