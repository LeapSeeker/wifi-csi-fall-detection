# 재부팅 후 빠른 환경 세팅

> **이 PC 특성**: 종료/재부팅하면 **D드라이브에 설치한 것 외에는 전부 초기화**된다.
> 다행히 개발에 필요한 핵심은 모두 D드라이브에 있으므로 **재설치는 거의 필요 없고**,
> 매번 초기화되는 **PATH·실행정책·venv 활성화만 다시** 해주면 된다.

---

## TL;DR — 재부팅 후 이것만

프로젝트 폴더에서 PowerShell을 열고 **점-소싱(`.` 한 칸 띄우고)** 으로 실행:

```powershell
cd D:\Project\LastProject\wifi-csi-fall-detection
. .\tools\env_bootstrap.ps1
```

끝. 스크립트가 PATH 추가 → 실행정책 Bypass → venv 활성화 → 검증까지 한 번에 한다.
정상이면 마지막에 이렇게 나온다:

```
git version 2.54.0.windows.1
Python 3.11.9
torch 2.6.0+cu124 | cuda True
  OK: venv == requirements.lock.txt (57 pkgs)
```

> ⚠️ 반드시 **점-소싱**(`. .\tools\env_bootstrap.ps1`). 그냥 `.\tools\env_bootstrap.ps1`로
> 실행하면 자식 프로세스에만 적용되고 현재 창의 PATH/venv에는 안 남는다.

---

## 영구(D) vs 초기화(C) 구분

| 항목 | 위치 | 재부팅 후 |
|---|---|---|
| 프로젝트 소스 | `D:\Project\LastProject\wifi-csi-fall-detection` | ✅ 유지 |
| 포터블 Git | `…\Git\cmd\git.exe` (2.54.0) | ✅ 유지 |
| Python 베이스 | `D:\Python311` (3.11.9) | ✅ 유지 |
| venv | `…\.venv` (→ `D:\Python311` 참조) | ✅ 유지 |
| git 사용자 정보 | `.git/config` (로컬, kimjg0930) | ✅ 유지 |
| git remote `origin` | `.git/config` → github.com/LeapSeeker/… | ✅ 유지 |
| **시스템 PATH** | 환경변수 | ❌ 초기화 → **매번 추가 필요** |
| **PowerShell 실행정책** | 사용자/머신 설정 | ❌ 초기화 → Activate.ps1 차단될 수 있음 |
| 전역 git config | `C:\Users\user\.gitconfig` | ❌ 초기화 (단, 이 repo는 로컬 config라 무관) |

핵심: **재설치 불필요. PATH·실행정책·활성화만 매 세션 다시.**

---

## 수동 절차 (스크립트 없이 / 디버깅용)

### 1) PATH에 포터블 Git + Python 추가 (세션 한정)
```powershell
$env:PATH = "D:\Project\LastProject\wifi-csi-fall-detection\Git\cmd;D:\Python311;D:\Python311\Scripts;$env:PATH"
```

### 2) venv 활성화
- **권장(활성화 없이)**: 풀 경로로 호출 — 실행정책·PATH 신경 안 써도 됨
  ```powershell
  & "D:\Project\LastProject\wifi-csi-fall-detection\.venv\Scripts\python.exe" your_script.py
  ```
- **활성화하려면**: 재부팅 직후 PowerShell 기본 실행정책이 `Restricted`면 `Activate.ps1`이 막힌다.
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force   # 이 창에서만
  .\.venv\Scripts\Activate.ps1
  ```
  (또는 cmd 창에서: `.venv\Scripts\activate.bat`)

### 3) 검증
```powershell
git --version                                            # 2.54.0.windows.1
python --version                                         # Python 3.11.9
python -c "import torch; print(torch.cuda.is_available())"  # True
python -m pip freeze | Measure-Object -Line              # 57 (lock과 일치해야 함)
```

---

## venv가 깨졌을 때만 (재생성)

PATH/활성화로 안 되고 import 에러 등이 나면 venv 재생성:
```powershell
D:\Python311\python.exe -m venv "D:\Project\LastProject\wifi-csi-fall-detection\.venv"
& "D:\Project\LastProject\wifi-csi-fall-detection\.venv\Scripts\python.exe" -m pip install -r requirements.lock.txt
```
- `requirements.lock.txt` = 정확한 버전 고정본(57패키지). 일반 `requirements.txt`보다 이걸 쓴다.
- `D:\Python311`이 사라졌다면 Python 3.11.9를 D드라이브에 다시 설치해야 한다(드물게 D가 손상된 경우).

---

## git 소스 제어 메모

- remote: `origin` → `https://github.com/LeapSeeker/wifi-csi-fall-detection.git`
- 사용자 정보는 **로컬 `.git/config`** 에 저장(`kimjg0930` / `kimjg0930@gmail.com`) → 재설정 불필요.
- 전역 git 설정이 필요하면 초기화되므로 다시:
  ```powershell
  git config --global user.name "kimjg0930"
  git config --global user.email "kimjg0930@gmail.com"
  ```

---

## 자동화 스크립트

`tools/env_bootstrap.ps1` — 위 1~3단계 + 패키지 무결성 검사(`pip freeze` vs `requirements.lock.txt`)를
한 번에 수행. 불일치 시 재설치 명령을, venv 없을 시 재생성 명령을 안내한다.
경로는 스크립트 자기 위치 기준으로 자동 산출하므로 폴더를 옮겨도 동작한다.
