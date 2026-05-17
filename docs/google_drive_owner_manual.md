# SafeSignal Google Drive 소유자 운영 매뉴얼

이 문서는 `SafeSignal_Dataset` Google Drive 폴더를 만든 사람이 팀원에게 공유하고,
수집/업로드 환경을 유지하기 위한 운영 절차이다.

민감 정보 원칙:

- Google OAuth token, refresh token은 절대 공유하지 않는다.
- `rclone config show` 출력은 채팅/문서/Git에 붙여넣지 않는다.
- `SafeSignal_Dataset` 폴더 ID는 Git 문서/코드에 직접 쓰지 않고 팀원에게 별도 전달한다.

---

## 1. Google Drive 폴더 구조

소유자 계정의 Google Drive에 아래 폴더를 유지한다.

```text
SafeSignal_Dataset/
├── README
├── E1/
│   ├── S01/
│   ├── S02/
│   └── S03/
├── E2/
├── E3/
└── E4/
```

`E?`, `S??` 폴더는 업로드 과정에서 자동 생성될 수 있으므로 미리 모두 만들 필요는 없다.

---

## 2. 팀원 공유 설정

1. Google Drive에서 `SafeSignal_Dataset` 폴더 우클릭
2. `공유` 선택
3. 팀원 Google 계정 추가
4. 권한은 `편집자(Editor)`로 설정

팀원은 각자 본인 Google 계정으로 rclone 인증을 진행한다.
소유자 계정을 공유하거나 대신 로그인하지 않는다.

---

## 3. 폴더 ID 전달

`SafeSignal_Dataset` 폴더 URL 예:

```text
https://drive.google.com/drive/folders/XXXXXXXXXXXXXXXXXXXXXXXX
```

여기서 `XXXXXXXXXXXXXXXXXXXXXXXX` 부분이 폴더 ID이다.

팀원에게는 아래처럼 별도 전달한다.

```text
SafeSignal_Dataset 폴더 ID: XXXXXXXXXXXXXXXXXXXXXXXX
```

이 ID는 Git 저장소 문서나 코드에는 넣지 않는다.

---

## 4. 팀원에게 안내할 rclone 핵심 설정

팀원은 `rclone config`에서 아래처럼 진행한다.

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

이렇게 설정하면 팀원의 `gdrive:` remote 루트가 곧 `SafeSignal_Dataset` 폴더가 된다.

확인 명령:

```bash
./.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe lsd gdrive:
```

`E1`, `E2`, `README` 등이 보이면 정상이다.

---

## 5. 팀원 수집 실행 환경변수

Git Bash:

```bash
export SAFESIGNAL_DRIVE_UPLOAD=1
export SAFESIGNAL_DRIVE_REMOTE="gdrive:"
export SAFESIGNAL_RCLONE_BIN="/c/Project/LastProject/wifi-csi-fall-detection/.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe"

python collect/collect_main.py
```

PowerShell:

```powershell
$env:SAFESIGNAL_DRIVE_UPLOAD="1"
$env:SAFESIGNAL_DRIVE_REMOTE="gdrive:"
$env:SAFESIGNAL_RCLONE_BIN="C:\Project\LastProject\wifi-csi-fall-detection\.local\rclone\rclone-v1.74.1-windows-amd64\rclone.exe"

python collect/collect_main.py
```

업로드 결과:

```text
E1_S01_A_WALK_T001.csv
→ gdrive:/E1/S01/E1_S01_A_WALK_T001.csv
```

---

## 6. 기존 데이터 정리

`SafeSignal_Dataset` 폴더 자체는 유지한다.

정리 가능:

- 잘못 올라간 CSV
- 중복 파일
- 임시 테스트 폴더
- 구조가 맞지 않는 하위 폴더

유지:

- `SafeSignal_Dataset/`
- `SafeSignal_Dataset/README`
- 정상 수집 CSV

기존 로컬 CSV를 폴더 구조에 맞춰 업로드하려면 팀 공용 매뉴얼의
“기존 수집 데이터 1회 업로드” Python 스니펫을 사용한다.

---

## 7. 토큰 노출 시 대응

누군가 `token: {...}` 또는 `refresh_token`을 공유했다면 즉시 아래 순서로 처리한다.

1. Google 계정 권한 페이지 접속: <https://myaccount.google.com/permissions>
2. `rclone` 권한 제거
3. 로컬 remote 삭제

```bash
./.local/rclone/rclone-v1.74.1-windows-amd64/rclone.exe config delete gdrive
```

4. `rclone config`를 다시 실행해 remote 재생성

---

## 8. 운영 체크리스트

수집 시작 전:

- [ ] 팀원이 `rclone lsd gdrive:`로 `SafeSignal_Dataset` 내부를 볼 수 있는지 확인
- [ ] `SAFESIGNAL_DRIVE_UPLOAD=1`
- [ ] `SAFESIGNAL_DRIVE_REMOTE=gdrive:`
- [ ] `SAFESIGNAL_RCLONE_BIN` 경로 확인

수집 종료 후:

- [ ] Google Drive에 `E?/S??/파일.csv` 구조로 업로드됐는지 확인
- [ ] `data/upload_log.md`에서 FAIL 항목 확인
- [ ] 실패 파일은 기존 데이터 업로드 스니펫으로 재업로드
