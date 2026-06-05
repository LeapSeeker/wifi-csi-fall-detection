# onset 검수 도구 사용법

1. `onset_review.html` 을 브라우저(Chrome/Edge/Firefox)로 더블클릭해 연다.
2. 키보드: A=승인 / C=제외 / M=수정 / ←→=이동. 또는 버튼 클릭.
   - 승인: 추천 onset 그대로 확정 → 자동 다음.
   - 제외: 사유 선택(no_clear_transient/beep_misfire/walking_residual/other) → 제외 확정.
   - 수정: 모달에서 onset frame 직접 입력(후보 클릭=자동입력) → 저장.
3. 매 판정마다 브라우저 localStorage 에 자동저장(닫았다 열어도 복구).
4. 다 끝나면 상단 **Export CSV** → `review_decisions.csv` 다운로드.
   그 파일을 `finalization/review_tool/` 에 두고 Claude Code 에게 "v2 생성" 요청.

## plot 이 안 보이면 (file:// 보안 차단 시)
같은 폴더 상위(finalization/)에서 로컬 서버를 띄우고 접속:
    cd debug/modeling/diag_out/onset_detector/finalization
    python -m http.server 8000
브라우저에서 http://localhost:8000/review_tool/onset_review.html

판정은 진규. 추천은 참고용. 원본/동결 파일 무수정.
