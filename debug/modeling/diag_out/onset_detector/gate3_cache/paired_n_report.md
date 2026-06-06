# 항목4 선행 — fixed ∩ onset_primary paired N 집계

paired = 동일 세션에서 fixed 생성 AND onset_primary 실제 생성(crop_out_of_bounds 제외). crop_index.csv 기준.

## 1) paired 전체 / 2) non-WALK / 3) WALK (split별)
구분 | train | val | test | train+val | 합계
---|---|---|---|---|---
전체 paired | 114 | 26 | 35 | 140 | 175
non-WALK paired | 82 | 19 | 27 | 101 | 128
WALK paired | 32 | 7 | 8 | 39 | 47

## 4) non-WALK subtype별 paired (split별)
subtype | train | val | test | train+val | 합계
---|---|---|---|---|---
SIT_F | 21 | 3 | 9 | 24 | 33
SIT_B | 22 | 4 | 5 | 26 | 31
STD_F | 18 | 6 | 8 | 24 | 32
STD_B | 21 | 6 | 5 | 27 | 32

## 5) onset_primary crop_out_of_bounds
- split별: train 53 / val 19 / test 16 (합 88)
- subtype별: SIT_B 9, SIT_F 13, STD_B 19, STD_F 19, WALK_B 11, WALK_F 17

## 6) 참고 — fixed ∩ onset_reduced paired (ablation)
구분 | train | val | test | train+val | 합계
---|---|---|---|---|---
전체 paired(reduced) | 146 | 39 | 48 | 185 | 233
non-WALK paired(reduced) | 105 | 28 | 37 | 133 | 170

## 판정 (non-WALK val/test 기준)
- non-WALK **val N=19** → 제한적 결론(CI 중심)
- non-WALK **test N=27** → main 비교 가능(20+)
- non-WALK train N=82 (학습 coverage 근거, 성능 N 아님)
- train+val=101 은 개발 coverage 근거로만(성능 결론 N 아님).

## reduced vs primary paired N
- non-WALK paired: primary train+val 101 vs reduced 133 → reduced가 더 많음 (OOB 차이 반영)
