# SafeSignal onset 작업 핸드오프 (2026-06-05 세션 종료 시점)

대화에만 있던 맥락을 다음 세션/다른 PC가 이어받도록 박아둔 메모. 상세 결정은 ai-workspace
STATE D-031(별도 repo), 수치 산출물은 같은 폴더 csv/json/md/plots 참조.

## 현재 위치
- **wifi-csi-fall-detection** repo, 브랜치 **`feature/event-centered-gate1`** (origin push 완료, commit `ab986ef`).
- **ai-workspace** = 별도 repo(LeapSeeker/ai-workspace), STATE D-031에 Gate 1·2·v1 결정 기록(push `ae16d2b`).
  - ai-workspace git identity는 로컬 config에 kimjg0930 설정됨(.git/config는 D드라이브라 유지).

## 재부팅 후 복구
```powershell
cd D:\Project\LastProject\wifi-csi-fall-detection
. .\tools\env_bootstrap.ps1
git checkout feature/event-centered-gate1   # 다른 PC면 git fetch 먼저
```

## 진행 상태 (Gate 1·2 확정, v1 done, 검수 pending)
- Gate 1: 좌표계 = **clean400_concat** 확정 (splice artifact 아님, WALK_B recall_gain +0.153).
- Gate 2: detector = **nominal[190:350] / k_mad=3.0 / sustain=5 / smooth=5**, soft = **set A**, broad=diagnostic only.
- onset_manifest **v1_auto_reviewed**: auto_reviewed 214 usable / pending_manual 135(WALK 68) / excluded 11.
  provisional onset median **132.5(~131)** — v2에서 재계산. WALK 과소대표(WALK_B usable 37%).

## 다음 할 일: 진규 onset 검수 (pending_manual)
도구: `review_tool/onset_review.html` (브라우저로 열기. plot 안 뜨면 README의 http.server).

### 검수 triage (priority_review_queue.csv 98개 train/val)
- **A) rise_not_found 27** — plot 필수, 신호 약/없음. **WALK_B rise_not_found 7개가 최난이도**(onset 본질적 모호).
- **B) beep/too_early 56** — queue의 `topk_cand_frames_clean`(in-window 후보)로 빠른 판정, 애매한 것만 plot.
- **C) soft-only 11** — valid rise 있음, 대체로 빠른 confirm(승인).
- **D) 기타 4** — 개별 확인.
- 권장 순서: C(워밍업) → B(후보 보고) → A(plot 정독, 배치 정책).

### ★ 미결정: WALK_B rise_not_found 배치 정책 (진규 결정 필요)
7개 전부 패턴 동일(걷기가 baseline[50:150] 점령 → threshold 폭등 → search 평탄, pob 1.11~1.28 약함).
- **정책1**: rise_not_found라도 search energy max 지점을 onset으로 채택(beep 아니고 clean 매핑 시). → 7개 살림, onset 신뢰 낮음.
- **정책2**: pob 낮으면(예: <1.3) 명확한 transient 없음으로 제외. → 7개 빠짐, WALK_B 더 줄어듦(이미 22/60).
- auto_exclude_candidate는 v1에서 0건(보수 기준). 즉 자동제외 안 했고 전부 진규 판단 대기.

### ★ 검수도구 주의 (중요)
- 판정은 **브라우저 localStorage(그 PC/브라우저 로컬)**에 저장됨. git에 안 올라감.
- **PC 바꾸거나 세션 끝내기 전 반드시 상단 [Export CSV]** → `review_decisions.csv` 저장/커밋.
  안 하면 학교에서 한 판정이 집에서 사라짐.

## v2 생성 흐름 (검수 후)
검수 끝 → Export `review_decisions.csv` → Claude Code에게 "v2 생성" 요청.
→ Claude가 `manifest_v1_auto_reviewed.csv` + `review_decisions.csv` 읽어 **manifest_v2_manual_augmented** 생성:
  - approve/modify → source=manual_confirmed, usable_for_training=true, onset 확정
  - exclude → source=excluded, exclude_reason 기록, usable=false
→ v2의 auto_reviewed+manual_confirmed 합쳐 **final onset median 재계산**(re-alignment point), 그 후 항목 4(onset-aligned crop) 진행.

## 미해결(구조적, 발표 후)
- WALK onset baseline contamination: baseline window가 걷기 잡음. 현재 처리=수동검수 수용.
  future options(보류): ① WALK 전용 baseline window ② WALK onset용 다른 신호.
- 항목 4 진행 시 주의: auto_reviewed가 WALK 과소대표라 onset-aligned 효과가 실제보다 좋게 보일 위험 → subtype별 분리 평가 + WALK는 v2 후 재평가.

## balanced 증강 실험 (2026-06-08 준비, 학교 PC 실행 대기)

목적: 기존 fall-only 더미의 class-ratio artifact 통제 — fall 더미 + **신규 non-fall 더미**(C/D 공유)를
추가해 6-class effective share 를 원본(A/B)으로 보존(A의 fixed/B의 onset target). 동결 train.py 무수정.

### 구현 완료 스크립트 (노트북 작성·검증, GPU 실행은 학교 PC)
- `debug/dummy_gen/balanced/nonfall_dummy_generate.py` — train non-fall raw window 재생성
  (preprocess_safesignal_file: rx=both/stride300/tail_window/max_gap100)→cache allclose 검증→
  perturb(6 profile, body×speed)→window_to_model_input. robust distance p97.5 gate + amp ratio[0.5,2.0].
- `debug/modeling/item4_build_balanced_caches.py` — base + 기존 fall 더미 crop + non-fall shared →
  item4_cache_{fixed,onset_primary}_balanced_aug.npz. assert: 더미 train-only/val·test 0/origin train/C·D nf set 동일.
- `debug/modeling/item4_train_balanced.py` — frozen split + 더미 온라인증강 skip + arm별 target_share/n_c
  class weight(effective share tol 1e-3) + runtime counter. A/B 재사용 점검(`--check-ab-reuse`)/재학습(`--retrain-ab`).
- `debug/modeling/item4_balanced_eval.py` — A/B/C/D, arm별 ckpt root(A/B=checkpoints_item4_reg→없으면 balanced),
  Primary D−B/Secondary C−A/Interaction, frontier(FAR≤0.15/0.20/0.30/maxF1), 판정(supported/directional/null),
  low_diversity 결론부 자동.

### 실행 순서 (학교 PC, GPU)
```
python debug/dummy_gen/balanced/nonfall_dummy_generate.py          # ① non-fall 더미
python debug/modeling/item4_build_balanced_caches.py              # ② balanced cache 2개
python debug/modeling/item4_train_balanced.py --smoke            # ③ smoke(preflight/share/counter/1ep)
python debug/modeling/item4_train_balanced.py --policies fixed_balanced_aug onset_primary_balanced_aug \
  --seeds 42 43 44 45 46 --epochs 40 --weight-decay 1e-4 --patience 8 --early-stop-start 8 \
  --ckpt-root checkpoints_item4_balanced                          # ④ 본학습 C/D
python debug/modeling/item4_balanced_eval.py                     # ⑤ 평가
```
A/B 재사용 불가 시(`--check-ab-reuse` 리포트 확인): `item4_train_balanced.py --retrain-ab --seeds ... --ckpt-root checkpoints_item4_balanced`.

### ★ preflight 주의 (스펙 0-1) — 없으면 fail-fast
- `debug/dummy_gen/out2/dummies_clean400.npz` **git에 없음(대용량 ignore)** → Drive/백업 복원 필수.
  lineage(696)/use(301) 정합 확인 후 진행. 복구 불가로 재생성 시 fall 더미 bit-identical 미보장 → report 명시.
- `debug/modeling/diag_out/onset_detector/item4/eval_windows.pkl` 도 ignore → 없으면
  `item4_precompute_eval_windows.py` 재생성 후 평가.

### 산출물 / Drive 업로드 대상 (대용량 = Drive, 추적은 스크립트·lineage·report 만)
- git 추적: 스크립트 4개, `debug/dummy_gen/balanced/nonfall_lineage.csv`, `*nonfall_quality_report*`,
  `debug/modeling/balanced_aug/*.json|md|csv`, 본 HANDOFF.
- Drive 업로드(ignore): `debug/dummy_gen/balanced/nonfall_dummies.npz`,
  `item4_cache_{fixed,onset_primary}_balanced_aug.npz`, `model/finetune/checkpoints_item4_balanced/`.

### 완료 조건 체크 (스펙 §10)
preflight 통과 / dummies_clean400 정합 / nonfall 더미 생성 / balanced cache 2개 / smoke 통과 /
C·D 5seed ckpt / dummy online aug 0 / val·test dummy 0 / effective share tol 1e-3 / eval report / HANDOFF.
범위 밖: slow v2, post-only/Q2, multi-window, threshold/test 튜닝.
