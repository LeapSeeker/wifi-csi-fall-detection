# SafeSignal 핸드오프 — 항목4 onset-aligned + 더미 증강 (2026-06-06)

다른 환경(팀원 집/Colab)에서 이어받기 위한 전수 핸드오프. **데이터=Drive, 코드/문서=git.**
브랜치 `feature/event-centered-gate1`. 상세 결정은 ai-workspace STATE D-031.

## 1. 현재 상태 (한눈에)
- **항목4 (onset-aligned crop) 완료**: onset_primary가 fixed 대비 **같은 recall에 FAR 유의 감소**
  (sealed test non-WALK paired N=27: ΔFAR −0.139 [CI −0.191,−0.091], Δrecall −0.015 ns). → "onset 정렬의 가치 = 오발화(FAR) 억제" 검증.
  운영점(frontier, 정규화): FAR≤0.15→recall0.77 / recall85%→FAR~0.20 / max-F1 0.71. 목표(recall≥85 AND FAR≤15) 동시 미달, F1≥0.85는 event-level(180:27) 구조상 불가.
- **더미 4-arm (A/B/C/D × 5seed) 완료**: 더미가 onset FAR을 강화 못함 — B→D **ΔFAR +0.11(악화)**, Δrecall ns.
  원인=class 불균형(fall:non 0.078→0.259, fall 3배). → **(B) class-ratio 통제 재검증** 진행.
- **(B) 통제 결과 (fall effective mass 원본 고정: fixed_aug 0.426/onset_aug 0.300)**:
  B→D ΔFAR **+0.110(유의) → +0.028 [−0.014,+0.071] 유의차 미검출**, 결합 (D−C)−(B−A) +0.118→+0.026(ns), Δrecall −0.044(ns).
  → **"더미가 onset FAR 악화"는 class 불균형 artifact 확정.** 통제 후 더미 효과 = **null(중립)** — 도움도 해도 안 됨(N=27 under-powered).
  D_ctrl: recall 0.874 / FAR 0.388 vs B 0.919 / 0.360. **결론: 현재 in-domain 더미(fall만)는 onset-aligned 성능을 강화하지 못함(중립).** 개선하려면 비낙상 포함 균형 증강 필요.

## 2. 더미 생성 맥락 (왜 이렇게 했나)
- **직접 생성**: 팀원 6/7.py 로직을 clean400 좌표로 이식. **factor/scale/crop_offset/snr 전부 lineage 저장**
  (팀원 원본은 난수 미저장이라 onset 역산 불가했음). 좌표=clean400(Gate1: raw[50:150]+[200:400]+[450:550]).
- **slow 제외**: slow time_warp가 onset 추적 약점(delta median 17 vs normal 5.5/fast 9.0). onset-aligned엔 normal+fast만.
- **이중 증강 fix**: train.py 온라인 증강(jitter/scale/timewarp/noise)이 더미에도 걸려 이중 증강 → **더미는 온라인 증강 skip**(filename 마커 식별), 원본만 유지. (item4_train.py monkeypatch)
- **2차 use 301**: onset 교차검증(expected 해석 vs detected Gate2 detector) 통과분. use onset_delta median2/p90 6, splice 경계 새 artifact 1%, baseline_noise 감소.
- **결론 범위 한정**: within-subject / in-domain augmentation 효과. 새 subject/env 일반화 해석 금지.

## 3. 핵심 발견 (정성)
- **WALK**: 제외 1위가 walking_residual 아니라 no_clear_transient — 낙상 transient 자체가 약/묻힘(걷기 가림 아님). onset-aligned 부적합(구조적).
- **SIT**: 완만한 낙상(스르륵) — 급격한 단발 없이 baseline 위로 완만히 솟음.
- **환경/피험자 품질**: 세션 단위 판정(일괄 금지). low_quality 9개 전부 noise>1.206(E2_S02/E4_S03), E1/E4_S01 미포함.
- **onset 정렬**: FAR 억제가 핵심 가치. recall은 이미 포화(~0.92).

## 4. 산출물 분류 (git / Drive / 제외)
### A. GitHub (가벼움, 추적)
- 스크립트: `debug/modeling/{build_gate3_cache,item4_build_policy_cache,item4_train,item4_precompute_eval_windows,item4_event_eval,item4_build_arm_caches,item4_arms_eval,build_manifest_v2,analyze_exclude_reasons,paired_n_report,sanity_gate_item4}.py`, `debug/dummy_gen/{dummy_clean400_lib,dummy_sanity,dummy_generate,dummy_validate}.py`
- 결과/리포트: `.../item4/results/`, `.../item4/arms_results/`, `gate3_cache/*.md|json|crop_index.csv`, `finalization/manifest_v2_*`, `review_tool*/review_decisions*.csv`, `dummy_gen/out*/lineage.csv`
- run config/history(.json), HANDOFF.md, STATE(ai-workspace)
### B. Google Drive (무거움) — **수동 업로드 필요**
> 자동 업로드 불가 2가지 이유: ① MCP create_file는 base64 인자라 100MB npz 전송 불가(payload), ② 보안 classifier가 private 데이터 외부반출을 하드 차단. → 사용자가 직접 Drive에 올려야 함.
> 생성된 Drive 폴더(top): **https://drive.google.com/drive/folders/1T5w4WSszqm7Pfsqf053Fe1Tq9Ra_ORoD** (`SafeSignal_handoff_20260606`). 여기에 아래 파일들을 수동 업로드(cache/dummy/ckpt 하위 폴더 만들어 정리 권장).
| 로컬 경로 | 용량 | 용도 |
|---|---|---|
| `model/pretrained/checkpoints/best.pt` | 1.5MB | fine-tune init |
| `model/pretrained/checkpoints/dataset_cache_e12_w300_s300_lag1_20_tail_ps.npz` | 16.6MB | Alsaify cache(combined sampler) |
| `model/finetune/cache/safesignal_e1234_pretrained6.npz` | 6.7MB | non-fall source |
| `debug/modeling/diag_out/onset_detector/item4/item4_cache_*.npz` | ~27MB(5개) | 4-arm 학습 cache(fixed/onset_primary/_reduced/_aug) |
| `debug/modeling/diag_out/onset_detector/item4/eval_windows.pkl` | 10MB | event eval sliding z-SDP |
| `debug/dummy_gen/out2/dummies_clean400.npz` | 98.6MB | 2차 더미(use 301 포함) |
| `debug/dummy_gen/out/dummies_clean400.npz` | 49.3MB | 1차 더미(slow 포함, fixed-only 후보) |
| `model/finetune/checkpoints_item4_arms*/` (.pt) | ~270MB | 학습 ckpt(선택 — seed 고정이라 재학습 가능) |
> ckpt는 seed 고정이라 재학습으로 복원 가능 → Drive 필수 아님. cache/dummy/pretrained가 핵심.
### C. 제외
- 팀원 `debug/dummy_gen/reference/` 6·7·8.py (무수정 참조용)
- 임시: `reports/tmp_*.npz`, `Inspect_demo_b.py`, `test.py`, `data/dummy/`(데모)
- `.venv/`, `Git/`(포터블), 로그

## 5. 팀원 집에서 이어받는 절차
1. `git pull` (브랜치 `feature/event-centered-gate1`) — 코드/스크립트/lineage/리포트/HANDOFF 확보.
2. Drive `SafeSignal_Dataset/handoff/` 에서 cache/dummy/pretrained 다운로드 → 동일 상대경로 배치
   (`model/pretrained/checkpoints/`, `model/finetune/cache/`, `debug/modeling/diag_out/onset_detector/item4/`, `debug/dummy_gen/out2/`).
3. 환경: `. .\tools\env_bootstrap.ps1`(이 PC) 또는 Colab에 requirements + GPU.
4. **재현/이어가기**:
   - 4-arm 재학습: `python debug/modeling/item4_train.py --policies fixed onset_primary fixed_aug onset_primary_aug --seeds 42 43 44 45 46 --ckpt-root checkpoints_item4_arms --weight-decay 1e-4 --patience 8 --early-stop-start 8 --epochs 40` (+`--control-fall-mass` for (B))
   - 평가: `ITEM4_CKPT_DIR=checkpoints_item4_arms python debug/modeling/item4_arms_eval.py`
   - cache 없으면 재생성: `item4_build_policy_cache.py` → `item4_build_arm_caches.py`(아 cache+더미), dummy는 `dummy_generate.py --speeds normal fast --out-subdir out2` (data/cleaned 필요).
5. data/cleaned(원본 수집 CSV) 없으면 팀 공용 데이터에서 확보 — 모든 cache의 입력.

## 6. 다음 후보 (개선 필요 시)
- **비낙상 더미로 균형 증강** (현재 fall만 늘려 FAR 교란 — (B)가 이걸 통제). 비낙상도 증강해 ratio 유지.
- multi-window (onset_primary train 114개=fixed 절반, 데이터 여지).
- 신규 세션 수집(새 subject/env 일반화 — 현재 within-subject 한정).

## 7. 재현성
- split: within_subject seed42 (frozen, manifest와 일치 검증). 학습 seed 42–46.
- 동일 cache + 동일 seed → 동일 결과. 동결 파이프라인 RPCA→ACF→SDP→z 무수정.
- non-WALK sealed test N=27 경계선 → seed 민감도 가능(5-seed mean±std + CI 보고).
