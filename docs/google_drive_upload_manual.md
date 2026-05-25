# SafeSignal Google Drive 데이터 업로드 매뉴얼

이 문서는 자체 수집 CSV를 Google Drive로 공유하기 위한 설정/운영 절차를 정리한다.

원칙:

- 수집 원본 CSV의 1차 저장 위치는 항상 로컬 `data/raw/`이다.
- Google Drive 업로드는 로컬 저장이 성공한 뒤 실행되는 보조 단계이다.
- 업로드가 실패해도 로컬 CSV는 삭제되지 않는다.
- API 키, OAuth 토큰, 개인 계정 정보는 저장소에 커밋하지 않는다.

---

## 1. 현재 구조

수집 프로그램 저장 흐름:

```text
collect/collect_main.py
→ SessionRecorder.save_session()
→ data/raw/E{env}_S{subj}_A_{activity}_T{trial}.csv 저장
→ Google Drive 자동 업로드(선택, E?/S?? 하위 폴더 자동 분류)
```

자동 업로드는 `rclone`을 사용한다. 기본값은 비활성화이며, 환경변수를 설정한 경우에만 실행된다.

원격 루트는 `gdrive:`처럼 데이터셋 루트 폴더 자체를 가리키도록 지정한다.
업로더가 파일명에서 환경/피험자를 파싱해 `E?/S??/` 하위 경로를 자동으로 붙인다.

---

## 2. 최초 1회 설치

### 2.1 rclone 설치

Windows 기준:

1. <https://rclone.org/downloads/> 에서 Windows용 rclone 다운로드
2. 압축 해제
3. `rclone.exe`가 있는 폴더를 PATH에 추가하거나, 실행 파일 경로를 따로 기록

설치 확인:

```powershell
rclone version
```

위 명령이 동작하면 설치 완료.

---

## 3. Google Drive remote 설정

최초 1회만 수행한다.

팀원은 각자 본인 Google 계정으로 로그인한다. 단, 업로드 대상은 팀장이 공유한
`SafeSignal_Dataset` 폴더로 고정한다. 이를 위해 rclone 설정 중
`root_folder_id`에 팀장이 공유한 폴더 ID를 입력한다.

```powershell
rclone config
```

아래 순서대로 입력한다.

```text
n
name> gdrive
```

그 다음 `Storage` 목록이 길게 나온다. 목록에서 `Google Drive` 항목 번호를 찾아 입력한다.
rclone 버전에 따라 번호가 달라질 수 있으므로 숫자를 외우지 말고 `Google Drive` 문구를 확인한다.

이후 질문은 아래처럼 진행한다.

```text
client_id> Enter
client_secret> Enter
scope> 1
root_folder_id> <팀장이 공유한 SafeSignal_Dataset 폴더 ID 입력>
service_account_file> Enter
Edit advanced config? n
Use web browser to automatically authenticate rclone with remote? y
```

브라우저가 열리면 본인 Google 계정으로 로그인하고 권한을 승인한다.

개인 Google Drive를 사용하는 경우:

```text
Configure this as a Shared Drive (Team Drive)? n
```

마지막 확인:

```text
Keep this "gdrive" remote? y
```

최종 메뉴로 돌아오면 종료한다.

```text
e/n/d/r/c/s/q> q
```

전체 입력 요약:

```text
n
name> gdrive
Storage> Google Drive 번호
client_id> Enter
client_secret> Enter
scope> 1
root_folder_id> <팀장이 공유한 SafeSignal_Dataset 폴더 ID 입력>
service_account_file> Enter
Edit advanced config? n
Use web browser to automatically authenticate? y
Shared Drive? n
Keep this remote? y
마지막 메뉴에서 q
```

주의:

- `Configuration complete` 화면에 `token: {...}` 이 보일 수 있다.
- 이 토큰은 Google 계정 접근 권한이므로 채팅/문서/Git에 공유하지 않는다.
- 토큰을 실수로 공유했다면 Google 계정 권한 페이지에서 rclone 권한을 제거하고 remote를 다시 만든다.
- 폴더 ID는 Git 문서/코드에 직접 넣지 않고, 팀장이 팀원에게 별도로 전달한다.

설정 확인:

```powershell
rclone lsd gdrive:
```

`E1`, `E2`, `README` 등 `SafeSignal_Dataset` 내부 항목이 보이면 성공.

---

## 4. 기존 수집 데이터 1회 업로드

이미 로컬 `data/raw/`에 있는 CSV는 자동 업로드 대상이 아니므로 한 번 직접 올린다.

권장 방식은 SafeSignal 업로더를 사용하는 것이다. 파일명에서 `E?`, `S??`를 읽어
Google Drive 하위 폴더를 자동 생성한다.

Git Bash:

```bash
export SAFESIGNAL_DRIVE_UPLOAD=1
export SAFESIGNAL_DRIVE_REMOTE="gdrive:"
export SAFESIGNAL_RCLONE_BIN="/c/Project/LastProject/wifi-csi-fall-detection/.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe"

python - <<'PY'
from pathlib import Path
from collect.drive_upload import upload_file

for path in sorted(Path("data/raw").glob("*.csv")):
    result = upload_file(path)
    print(path.name, "OK" if result.success else f"FAIL: {result.message}")
PY
```

PowerShell:

```powershell
$env:SAFESIGNAL_DRIVE_UPLOAD="1"
$env:SAFESIGNAL_DRIVE_REMOTE="gdrive:"
$env:SAFESIGNAL_RCLONE_BIN="C:\Project\LastProject\wifi-csi-fall-detection\.local\rclone\rclone-v1.74.1-windows-amd64\rclone.exe"

@'
from pathlib import Path
from collect.drive_upload import upload_file

for path in sorted(Path("data/raw").glob("*.csv")):
    result = upload_file(path)
    print(path.name, "OK" if result.success else f"FAIL: {result.message}")
'@ | python -
```

업로드 결과 구조:

```text
gdrive:
└── E1/
    └── S01/
        └── E1_S01_A_WALK_T001.csv
```

단순히 한 폴더에만 올려도 되는 경우에는 rclone을 직접 사용할 수 있다.
다만 아래 명령은 `E?/S??` 하위 폴더 자동 분류를 하지 않는다.

```powershell
cd C:\Project\LastProject\wifi-csi-fall-detection
rclone copy data\raw gdrive:
```

업로드 확인:

```powershell
rclone ls gdrive:
```

주의:

- `rclone copy`는 로컬 파일을 삭제하지 않는다.
- 같은 파일명이 이미 있으면 크기/수정시간 기준으로 필요한 파일만 전송한다.

---

## 5. 새 터미널에서 서버 UI 수집 + 자동 업로드

새 Git Bash 터미널을 열 때마다 아래 순서로 진행한다.

### 5.1 가상환경 활성화

```bash
cd /c/Project/LastProject/wifi-csi-fall-detection
source .venv/Scripts/activate
```

### 5.2 Drive 대상 폴더 확인

```bash
./.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe lsd gdrive:
```

`SafeSignal_Dataset` 폴더가 보이면 다음 단계로 진행한다.

### 5.3 자동 업로드 환경변수 설정

```bash
export SAFESIGNAL_DRIVE_UPLOAD=1
export SAFESIGNAL_DRIVE_REMOTE="gdrive:SafeSignal_Dataset"
export SAFESIGNAL_RCLONE_BIN="/c/Project/LastProject/wifi-csi-fall-detection/.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe"
```

같은 터미널 창에서는 서버를 껐다 켜도 이 값이 유지된다. 새 터미널을 열면 다시 설정해야 한다.

설정 확인:

```bash
echo $SAFESIGNAL_DRIVE_REMOTE
```

정상 출력:

```text
gdrive:SafeSignal_Dataset
```

### 5.4 서버 실행

```bash
cd /c/Project/LastProject/wifi-csi-fall-detection/server
python main.py
```

대시보드에서 수집 탭으로 들어가 `E`, `S`, activity를 선택해 수집한다.

### 5.5 업로드 확인

수집 저장 후 새 Git Bash 탭 또는 서버를 종료한 터미널에서 확인한다.

```bash
cd /c/Project/LastProject/wifi-csi-fall-detection
tail -20 data/upload_log.md
./.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe ls gdrive:SafeSignal_Dataset/E2/S02
```

동작 예:

```text
저장됨: data/raw/E1_S01_A_WALK_T003.csv
[Drive] 업로드 시작: E1_S01_A_WALK_T003.csv
[Drive] 업로드 완료: gdrive:SafeSignal_Dataset/E1/S01/E1_S01_A_WALK_T003.csv
```

환경변수를 설정하지 않으면 기존처럼 로컬 저장만 수행한다.

```text
[Drive] 자동 업로드 비활성화 (SAFESIGNAL_DRIVE_UPLOAD=1 로 활성화)
```

---

## 6. rclone이 PATH에 없을 때

`rclone.exe`가 PATH에 없다면 실행 파일 경로를 환경변수로 지정한다.

예:

```powershell
$env:SAFESIGNAL_DRIVE_UPLOAD="1"
$env:SAFESIGNAL_DRIVE_REMOTE="gdrive:SafeSignal_Dataset"
$env:SAFESIGNAL_RCLONE_BIN="C:\Tools\rclone\rclone.exe"
python collect/collect_main.py
```

---

## 7. 업로드 타임아웃 조정

기본 업로드 타임아웃은 120초이다. 네트워크가 느리면 늘릴 수 있다.

```powershell
$env:SAFESIGNAL_DRIVE_TIMEOUT_SEC="300"
```

---

## 8. 업로드 로그

업로드 성공/실패는 로컬 `data/upload_log.md`에 기록된다.

예:

```markdown
| time | status | local_path | remote_path | message |
|---|---|---|---|---|
| 2026-05-18 10:00:00 | OK | data\raw\E1_S01_A_WALK_T003.csv | gdrive:/E1/S01/E1_S01_A_WALK_T003.csv | uploaded |
```

`data/upload_log.md`는 개인 실행 로그이므로 Git에 커밋하지 않는다.

---

## 9. 실패 시 처리

### 9.1 `rclone not found`

원인:

- rclone이 설치되지 않았거나 PATH에 없음

해결:

```powershell
rclone version
```

이 명령이 실패하면 rclone 설치/PATH 설정을 다시 확인한다.
PATH를 수정하기 어렵다면 `SAFESIGNAL_RCLONE_BIN`을 사용한다.

### 9.2 `SAFESIGNAL_DRIVE_REMOTE is not set`

원인:

- 자동 업로드는 켰지만 remote 경로를 설정하지 않음

해결:

```powershell
$env:SAFESIGNAL_DRIVE_REMOTE="gdrive:"
```

### 9.3 Google 로그인/권한 문제

확인:

```powershell
rclone lsd gdrive:
```

실패하면:

```powershell
rclone config reconnect gdrive:
```

### 9.4 자동 업로드 실패한 파일 재업로드

로컬 CSV는 남아 있으므로 수동으로 다시 업로드한다.

```powershell
rclone copy data\raw gdrive:
```

특정 파일 1개만 다시 올릴 경우:

```powershell
rclone copyto data\raw\E1_S01_A_WALK_T003.csv gdrive:/E1/S01/E1_S01_A_WALK_T003.csv
```

---

## 10. 팀 운영 권장 방식

수집 시작 전:

```powershell
cd C:\Project\LastProject\wifi-csi-fall-detection
rclone lsd gdrive:
$env:SAFESIGNAL_DRIVE_UPLOAD="1"
$env:SAFESIGNAL_DRIVE_REMOTE="gdrive:"
python collect/collect_main.py
```

수집 종료 후:

```powershell
rclone copy data\raw gdrive:
```

종료 후 기존 데이터 1회 업로드의 Python 스니펫을 한 번 더 실행하면 자동 업로드 실패분까지
`E?/S??` 폴더 구조로 보정할 수 있다.

---

## 11. 주의사항

- Google Drive에 올라간 CSV와 로컬 CSV의 파일명이 같아야 한다.
- raw CSV는 전처리 변경과 무관하게 원본 데이터로 보존한다.
- Google Drive 업로드 설정이 되어 있어도 로컬 `data/raw/`는 삭제하지 않는다.
- 토큰/계정 정보가 들어 있는 rclone 설정 파일은 Git에 커밋하지 않는다.

---

## 12. Git Bash `drive` alias (수집 서버 + 자동 업로드 한 번에 실행)

5장의 절차(레포 이동 → 환경변수 설정 → Drive 연결 확인 → 서버 실행)를 매번 손으로
입력하지 않도록 Git Bash alias 하나로 묶는다.

> 이 wrapper는 **코드에 넣지 않는다.** `server/main.py`나 `tools/safesignal_debug.py`에
> collect-server wrapper를 추가하지 말고, 아래처럼 셸 alias로만 관리한다.

`drive` alias 동작:

1. `wifi-csi-fall-detection` 레포로 이동
2. `SAFESIGNAL_DRIVE_UPLOAD=1` 설정
3. `SAFESIGNAL_DRIVE_REMOTE=gdrive:SafeSignal_Dataset` 설정
4. `SAFESIGNAL_RCLONE_BIN`을 레포 내부 `rclone.exe`로 설정
5. `rclone lsd gdrive:` 로 Drive 연결 확인
6. `python server/main.py` 실행

### 12.1 일시 등록 (현재 터미널 창에서만)

새 Git Bash 창을 연 직후, 아래 한 줄을 붙여 넣으면 그 창에서만 `drive`를 쓸 수 있다.
창을 닫으면 사라진다.

```bash
alias drive='cd /c/Project/LastProject/wifi-csi-fall-detection && export SAFESIGNAL_DRIVE_UPLOAD=1 && export SAFESIGNAL_DRIVE_REMOTE="gdrive:SafeSignal_Dataset" && export SAFESIGNAL_RCLONE_BIN="/c/Project/LastProject/wifi-csi-fall-detection/.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe" && "$SAFESIGNAL_RCLONE_BIN" lsd gdrive: && python server/main.py'
```

등록 후 실행:

```bash
drive
```

`rclone lsd gdrive:` 에서 `SafeSignal_Dataset` 등 Drive 항목이 출력되면 연결 정상이고,
이어서 수집 서버가 뜬다. Drive 연결에 실패하면 `&&` 체인이 거기서 멈추므로
서버가 실행되지 않는다 — 이 경우 9장(실패 시 처리)을 따른다.

### 12.2 영구 등록 (`~/.bashrc`)

매번 붙여 넣기 싫으면 `~/.bashrc`에 한 번만 추가한다.

```bash
cat >> ~/.bashrc <<'EOF'

# SafeSignal: 수집 서버 + Drive 자동 업로드
alias drive='cd /c/Project/LastProject/wifi-csi-fall-detection && export SAFESIGNAL_DRIVE_UPLOAD=1 && export SAFESIGNAL_DRIVE_REMOTE="gdrive:SafeSignal_Dataset" && export SAFESIGNAL_RCLONE_BIN="/c/Project/LastProject/wifi-csi-fall-detection/.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe" && "$SAFESIGNAL_RCLONE_BIN" lsd gdrive: && python server/main.py'
EOF
```

현재 창에 즉시 반영(또는 Git Bash를 새로 열기):

```bash
source ~/.bashrc
```

이후로는 어느 Git Bash 창에서든 `drive` 한 번이면 된다.

### 12.3 참고

- 가상환경(`.venv`)을 쓰는 경우, alias 마지막의 `python`이 가상환경 파이썬을 가리키도록
  alias 실행 전에 `source .venv/Scripts/activate`를 먼저 하거나,
  alias 안의 `python`을 `.venv/Scripts/python.exe`로 바꿔도 된다.
- 레포 경로나 rclone 버전 폴더명이 바뀌면 위 경로 두 곳
  (`cd` 대상과 `SAFESIGNAL_RCLONE_BIN`)을 함께 수정한다.
- remote 루트를 `gdrive:` 자체로 쓰고 싶으면 `SAFESIGNAL_DRIVE_REMOTE` 값만 바꾼다.
  (`E?/S??` 하위 분류는 업로더가 파일명으로 자동 처리한다 — 1장 참고.)

---

## 13. Git Bash `train` alias (수집용 서버 = 추론 비활성 + 자동 업로드)

수집 품질 측정 중에는 RPCA/추론 부하가 UDP 수신/페어링/대시보드 갱신에 영향을 주고
`InferenceWorker input_queue full/drop` 로그가 계속 쌓인다. 이를 막기 위해 추론
프로세스를 아예 띄우지 않는 "수집용 서버" 실행을 `train` alias로 묶는다.

`drive` alias와 거의 같지만, **`SAFESIGNAL_DISABLE_INFERENCE=1`을 추가**하여
`server/main.py`가 `InferenceWorker`를 생성/start하지 않도록 한다. 서버 시작 로그에
`[Inference] disabled by SAFESIGNAL_DISABLE_INFERENCE=1`이 출력되면 정상이다.

> `drive`(추론 활성, 기존 동작)와 `train`(추론 비활성, 수집 전용)은 별도 alias로
> 공존한다. 수집만 할 때는 `train`, 실시간 추론까지 확인하려면 `drive`를 쓴다.

`train` alias 동작:

1. `wifi-csi-fall-detection` 레포로 이동
2. `SAFESIGNAL_DISABLE_INFERENCE=1` 설정 (추론 프로세스 미시작)
3. `SAFESIGNAL_DRIVE_UPLOAD=1` 설정
4. `SAFESIGNAL_DRIVE_REMOTE=gdrive:SafeSignal_Dataset` 설정
5. `SAFESIGNAL_RCLONE_BIN`을 레포 내부 `rclone.exe`로 설정
6. `rclone lsd gdrive:` 로 Drive 연결 확인
7. `python server/main.py` 실행

### 13.1 일시 등록 (현재 터미널 창에서만)

```bash
alias train='cd /c/Project/LastProject/wifi-csi-fall-detection && export SAFESIGNAL_DISABLE_INFERENCE=1 && export SAFESIGNAL_DRIVE_UPLOAD=1 && export SAFESIGNAL_DRIVE_REMOTE="gdrive:SafeSignal_Dataset" && export SAFESIGNAL_RCLONE_BIN="/c/Project/LastProject/wifi-csi-fall-detection/.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe" && "$SAFESIGNAL_RCLONE_BIN" lsd gdrive: && python server/main.py'
```

등록 후 실행:

```bash
train
```

### 13.2 영구 등록 (`~/.bashrc`)

```bash
cat >> ~/.bashrc <<'EOF'

# SafeSignal train alias start
alias train='cd /c/Project/LastProject/wifi-csi-fall-detection && export SAFESIGNAL_DISABLE_INFERENCE=1 && export SAFESIGNAL_DRIVE_UPLOAD=1 && export SAFESIGNAL_DRIVE_REMOTE="gdrive:SafeSignal_Dataset" && export SAFESIGNAL_RCLONE_BIN="/c/Project/LastProject/wifi-csi-fall-detection/.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe" && "$SAFESIGNAL_RCLONE_BIN" lsd gdrive: && python server/main.py'
# SafeSignal train alias end
EOF
```

현재 창에 즉시 반영(또는 Git Bash를 새로 열기):

```bash
source ~/.bashrc
```

등록 확인:

```bash
bash -lc "source ~/.bashrc && alias train"
```

### 13.3 참고

- `train`은 추론을 끄므로 낙상 알림/SMS/Pi4 fall 이벤트는 발생하지 않는다. 순수
  데이터 수집·저장·Drive 업로드 전용이다. 추론까지 필요하면 `drive`를 쓴다.
- `SAFESIGNAL_DISABLE_INFERENCE` 허용값은 `1`, `true`, `yes`, `on` (대소문자 무관).
  미설정/그 외 값이면 기존대로 추론이 활성화된다.
- `.env`에 `SAFESIGNAL_DISABLE_INFERENCE=1`을 넣어도 동작한다(`load_dotenv()` 이후 평가).
- 가상환경(`.venv`) 사용 시 alias 실행 전 `source .venv/Scripts/activate`를 먼저
  하거나, alias 안의 `python`을 `.venv/Scripts/python.exe`로 바꾼다.
