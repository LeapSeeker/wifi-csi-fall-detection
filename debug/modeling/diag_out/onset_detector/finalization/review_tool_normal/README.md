# onset 검수 도구 (normal-priority 17개, v2 인터랙티브 차트)

priority-high 98개 검수 후, manifest_v1 의 normal-priority train/val pending 17개 추가 검수용.
기존 98개 도구(review_tool/)와 **별도** — localStorage 키(onset_review_decisions_normal_v1)와
Export 파일명(review_decisions_normal.csv)이 분리되어 98개 판정과 섞이지 않음.

1. `onset_review.html` 을 브라우저로 더블클릭(곡선 임베드라 file:// 에서도 표시).
2. 차트: 파란 굵은 선=sparse energy(판단 기준). 구간 빗금/글자라벨, x눈금 1·10·100 차등,
   hover frame·energy, 클릭=선택 후보(초록선) → M 으로 수정 저장. (98개 도구와 동일 형식)
3. A=승인 / C=제외 / M=수정 / ←→=이동. 제외 사유 드롭다운은 4종 —
   Codex 5-primary(walking_residual/beep_misfire/no_clear_transient/low_quality_env_subject/
   data_short_or_corrupt) 재분류는 export 후 baseline_noise_ratio 교차참조로 별도 수행.
4. 끝나면 Export CSV → review_decisions_normal.csv → finalization/review_tool_normal/ 에 두고
   Claude Code 에 v2 manifest 생성 요청(98 + 17 통합).

판정은 진규. 추천은 참고용. 원본/동결 파일 무수정.
