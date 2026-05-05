### data 폴더 구조 안내

이 폴더는 학습/평가에 사용되는 모든 데이터의 보관 위치입니다.
실제 데이터 파일(.csv 등)은 .gitignore 로 제외되며, 폴더 구조만 git 에 보존됩니다.

1. 폴더 구성
data/
├── raw/                # 자체 수집 ESP32 CSI 원본 (.csv)
├── augmented/          # 증강 데이터 (.csv)
├── alsaify-raw/        # Alsaify 사전학습 데이터셋 (외부 다운로드, .gitignore 처리)
└── collection_log.md   # 수집 기록 (수집 시 한 행씩 추가)

2. Alsaify 사전학습 데이터 배치
노션에 등록된 링크에서 데이터셋 다운로드 후 압축 해제하여 아래 경로에 배치
data/
└── alsaify-raw/
    ├── Environment 1/
    │   ├── Subject_1/
    │   │   └── *.csv
    │   └── Subject_2/ ...
    └── Environment 2/
        ├── Subject_1/ ...
        └── ...

주의: Subject_*.zip 형태로 되어 있으면 반드시 압축 해제 후 배치해야 합니다.
주의: alsaify-raw/ 전체 폴더는 .gitignore 처리되어 있으므로 로컬에만 존재합니다.
참고: 사전학습 캐시(model/pretrained/checkpoints/dataset_cache*.npz)는 git 에
      포함되어 있어 raw CSV 없이도 재학습 가능 (--rebuild_cache 미지정 시).

3. 자체 수집 데이터 배치
ESP32 수집 결과물은 data/raw/ 에 .csv 형식으로 저장하고,
증강 결과는 data/augmented/ 에 저장합니다.

수집 기록은 data/collection_log.md 에 다음 양식으로 작성 (model/CLAUDE.md 준수):
| 날짜 | 피험자 | 동작 | 방향 | 횟수 | 환경 | 파일명 |

수집 목표: 240세션 (model/CLAUDE.md 「데이터 수집 전략」참조)
- 낙상(앉다가/서다가) 60, 걷기/앉기일어서기/눕기/제자리서기/달리기/물건집기 각 30
- 증강 5배 → 약 1,200 샘플

4. .gitignore 동작 요약
- data/raw/*.csv       → 원본 .csv 는 추적 제외
- data/augmented/*.csv → 증강 .csv 는 추적 제외
- data/alsaify-raw/    → 폴더 전체 추적 제외
폴더 자체는 .gitkeep 으로 보존되어 다른 팀원이 clone 시 동일한 구조를 받습니다.
