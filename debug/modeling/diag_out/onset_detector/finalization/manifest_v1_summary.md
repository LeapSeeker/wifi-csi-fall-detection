# onset_manifest v1 (auto_reviewed clean subset) 요약

total 360 | auto_reviewed 214(59.4%) pending_manual 135(37.5%) excluded 11 | usable_for_training 214(59.4%)
auto_exclude_candidate(no_clear_transient): 0

## subtype별 auto_reviewed / pending_manual / usable
subtype | auto | pending | usable | usable%
FALL_SIT_B | 35 | 23 | 35 | 58%
FALL_SIT_F | 42 | 14 | 42 | 70%
FALL_STD_B | 46 | 14 | 46 | 77%
FALL_STD_F | 42 | 16 | 42 | 70%
FALL_WALK_B | 22 | 37 | 22 | 37%
FALL_WALK_F | 27 | 31 | 27 | 45%

## split별 usable (test는 기록만, 기준변경 미사용)
- train: usable 128/230 (56%)
- val: usable 35/57 (61%)
- test: usable 51/72 (71%)
- out_of_scope: usable 0/1 (0%)

## provisional onset median (auto_reviewed usable only, clean frame)
- overall: 132.5
- by_subtype: {'FALL_SIT_B': 130.0, 'FALL_SIT_F': 132.0, 'FALL_STD_B': 142.0, 'FALL_STD_F': 135.5, 'FALL_WALK_B': 117.0, 'FALL_WALK_F': 145.0}
- by_split: {'train': 127.5, 'val': 143.0, 'test': 134.0, 'out_of_scope': None}
- auto_reviewed count by_subtype: {'FALL_SIT_B': 35, 'FALL_SIT_F': 42, 'FALL_STD_B': 46, 'FALL_STD_F': 42, 'FALL_WALK_B': 22, 'FALL_WALK_F': 27}

## priority review queue (high pending_manual)
- total 115 | train/val 98 | WALK_B 37 WALK_F 25 | plots 98
- 위치: D:\Project\LastProject\wifi-csi-fall-detection\debug\modeling\diag_out\onset_detector\finalization/priority_review_queue.csv, plots_priority/

## pending_manual soft_count 분포
- {0: 44, 1: 33, 2: 45, 3: 10, 4: 3, 5: 0}
