# onset 검수 도구 v2 사용법 (인터랙티브 차트)

1. `onset_review.html` 을 브라우저(Chrome/Edge/Firefox)로 더블클릭해 연다.
   - v2 는 곡선을 JS 로 직접 그리므로(PNG 아님) **곡선은 file:// 에서도 보인다.**
   - 곡선 데이터는 HTML 안에 임베드됨(energy_curves.json 빌드시 포함).
2. 차트 보는 법:
   - **파란 굵은 선 = sparse energy(판단 기준).** 낙상처럼 확 솟는 지점이 onset.
   - 회색 점선 thr / 주황 점선 auto rise = 참고용.
   - 구간: BASELINE(빗금)·SEARCH(파란빗금, 유효 onset 구간)·BEEP(주황 교차빗금) — 글자 라벨 병기.
   - x축 눈금: 1(작은선)·10(중간+숫자)·100(굵은+숫자) 3단계.
   - 마우스 올리면 frame·energy 표시. 클릭하면 '선택 후보'(초록선) → M 키로 그 값 수정 저장.
3. 키보드: A=승인 / C=제외 / M=수정 / ←→=이동. 또는 버튼.
   - 승인: 추천 onset 그대로 확정 → 자동 다음. (rise_not_found 세션은 추천 없음 → 수정/제외)
   - 제외: 사유 선택 → 제외 확정.
   - 수정: 모달에서 그래프 클릭/후보 클릭/직접 입력 → **초록 세로선 실시간 이동** → 저장.
     SEARCH 밖 frame 도 지정 가능(경고만, 입력 막지 않음).
4. 매 판정마다 localStorage 자동저장(닫았다 열어도 복구).
5. 끝나면 상단 **Export CSV** → `review_decisions.csv` 다운로드.
   그 파일을 `finalization/review_tool/` 에 두고 Claude Code 에게 "v2 생성" 요청.

## 곡선이 안 보이면
- 곡선은 HTML 임베드라 보통 file:// 더블클릭으로 충분.
- 만약 특정 세션이 'PNG 폴백'으로 뜨면 energy_curves.json 에 그 세션 곡선이 없는 것 →
  PNG(../plots_priority/) 참조. 로컬 서버가 필요하면:
    cd debug/modeling/diag_out/onset_detector/finalization
    python -m http.server 8000
  브라우저에서 http://localhost:8000/review_tool/onset_review.html

판정은 진규. 추천은 참고용. 원본/동결 파일 무수정.
