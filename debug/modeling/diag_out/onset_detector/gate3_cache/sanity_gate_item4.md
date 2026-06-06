# 항목4 Sanity gate 결과 (read-only)

## 1. cache class label 분포 (fall-only 기대)
policy | N | y 분포 | classes[fall_idx]
---|---|---|---
fixed | 349 | {0: 349} | fall
onset_primary | 175 | {0: 175} | fall
onset_reduced | 233 | {0: 233} | fall

## 2. split leakage (같은 filename 이 복수 split)
- 복수 split filename: **0** (없음 ✓)
## 3. cache split == manifest split (일관)
- 불일치: **0** (없음 ✓)

split 분포:
- fixed: {'test': 71, 'train': 223, 'val': 55}
- onset_primary: {'test': 35, 'train': 114, 'val': 26}
- onset_reduced: {'test': 48, 'train': 146, 'val': 39}

## 4. paired key 매칭 (fixed ∩ onset_primary, session 단위)
- 1세션 1crop: fixed dup 0, onset_primary dup 0 ✓
- onset_primary ⊆ fixed: 미포함 0 (전부 fixed에 존재 ✓)
- onset_reduced ⊆ fixed: 미포함 0 ✓
- paired(fixed∩onset_primary) = **175** (= onset_primary 175 와 일치 ✓)
- paired subtype/split 동일: 불일치 0 ✓

## 5. crop 범위 무결성 (길이300, [0,400], 정렬 일치, clamp 없음)
- 위반: **0** (없음 ✓)

## 6. X 텐서 무결성
- fixed: shape (349, 1, 28, 20) ✓ | finite True
- onset_primary: shape (175, 1, 28, 20) ✓ | finite True
- onset_reduced: shape (233, 1, 28, 20) ✓ | finite True

## 7. seed/config 재현성 기록
- split_id (동결): `within_subject_seed42_val0.2_test0.2_pretrained6`
- train seed (run config 고정 필요): **42**
- config: {'class_policy': 'pretrained6', 'pipeline': 'RPCA→ACF→SDP→global z-score', 'window': 300, 'clean_coord': 'clean400 concat (Gate1)', 'fall_label_index': 0}
- 동일 cache + 동일 seed → 동일 결과 재현 보장 조건 기록 완료.
- ⚠ non-WALK sealed test N=27 경계선 → seed 민감도 가능. 가능하면 3~5 seed multi-seed, 단일 seed면 seed 값 명시 + residual risk 기록.

## 종합 판정
check | 결과 | 비고
---|---|---
1.label fall-only | PASS ✓ | 모든 crop y=0(fall), classes[0]=fall
2.no split leakage | PASS ✓ | 0 sessions in >1 split
3.cache/manifest split 일치 | PASS ✓ | 0 mismatches
4.paired key 매칭 | PASS ✓ | dup 0/0, prim⊄fixed 0
4b.paired subtype/split 동일 | PASS ✓ | 0 mismatch
5.crop 범위 무결성 | PASS ✓ | 0 violations
6.X 무결성 | PASS ✓ | shape (N,1,28,20), no NaN/inf
7.재현성 기록 | PASS ✓ | seed/config recorded

### ✅ SANITY GATE PASS — GPU 학습 진행 가능
