# 게이트 2 onset detector probe 요약

split: within_subject_seed42_val0.2_test0.2_pretrained6 (cache safesignal_e1234_pretrained6.npz)
세션: total 360 | train=230 val=57 test=72 out_of_scope=1 | excluded<550=11
subtype별: {'FALL_SIT_B': 60, 'FALL_SIT_F': 60, 'FALL_STD_B': 60, 'FALL_STD_F': 60, 'FALL_WALK_B': 60, 'FALL_WALK_F': 60}

## onset 분포 (clean frame, nominal onset=100)
- rise_frame_clean: {'n': 2788, 'median': 132.0, 'p25': 114.0, 'p75': 157.0, 'min': 100.0, 'max': 262.0}
- peak_frame_clean: {'n': 3714, 'median': 182.0, 'p25': 157.0, 'p75': 207.0, 'min': 100.0, 'max': 299.0}
## search 비교
- rise_outside_nominal_rate: 0.08691499522445081
- broad_only_session_rate: 0.011111111111111112
- nominal_broad_delta: {'n': 3852, 'median': 0.0, 'p25': 0.0, 'p75': 0.0, 'min': -92.0, 'max': 0.0}
## param 안정성
- param_sensitivity_range_overall: {'n': 336, 'median': 10.0, 'p25': 1.0, 'p75': 33.0, 'min': 0.0, 'max': 139.0}
## needs_review
- hard reason 건수(param row): {'topk_no_consensus_hard': 114, 'frame_mapping_failed': 1082, 'rise_in_beep_region': 1082, 'rise_too_early_original_lt_200': 1082, 'rise_not_found': 318, 'peak_in_beep_region': 156, 'peak_outside_expected_fall_original_200_400': 156}
- any_hard 세션: 195 / priority high 세션: 213
- high_by_subtype: {'FALL_SIT_B': 29, 'FALL_SIT_F': 25, 'FALL_STD_B': 24, 'FALL_STD_F': 33, 'FALL_WALK_B': 60, 'FALL_WALK_F': 42}
## baseline 품질
- noise_ratio: {'n': 4188, 'median': 1.100820871585909, 'p25': 1.082482863708528, 'p75': 1.1278101577503528, 'min': 1.0517964746652875, 'max': 3.0572029460146517}
- mad_ratio: {'n': 4188, 'median': 0.05203764732410667, 'p25': 0.04416520295639803, 'p75': 0.06504443891219858, 'min': 0.02834284225358024, 'max': 0.184650900992475}
## soft 구성요소 raw 분포 (threshold 미확정)
- rise_strength: {'n': 3870, 'median': 1.2004821140158226, 'p25': 1.1587449882402387, 'p75': 1.2562363327017285, 'min': 1.0764652262060401, 'max': 5.021730889832063}
- rise_slope: {'n': 3870, 'median': 0.03646806363376594, 'p25': 0.02285450534982004, 'p75': 0.058777933896003595, 'min': -0.14663270969175082, 'max': 0.642021792712759}
- topk_spread: {'n': 3870, 'median': 2.0, 'p25': 2.0, 'p75': 3.0, 'min': 0.0, 'max': 179.0}
- confidence_ref: {'n': 3714, 'median': 0.5895425171273684, 'p25': 0.5580254889863855, 'p75': 0.6202076206140255, 'min': 0.2436944997481294, 'max': 0.8702320736853993}
