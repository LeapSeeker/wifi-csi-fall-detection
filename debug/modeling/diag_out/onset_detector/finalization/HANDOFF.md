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
