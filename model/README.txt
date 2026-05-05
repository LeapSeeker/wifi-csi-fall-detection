### 실행 방법
1. 코드 받기
# GitHub에서 직접 clone
git clone https://github.com/LeapSeeker/wifi-csi-fall-detection
cd wifi-csi-fall-detection
git checkout feature/pretrained-model

2. Alsaify 데이터 전송
노션에 등록된 링크에서 데이터셋 다운로드 후 압축 해제하여 아래 경로에 배치
wifi-csi-fall-detection/
└── data/
    └── alsaify-raw/
        ├── Environment 1/
        │   ├── Subject_1/
        │   │   └── *.csv
        │   └── Subject_2/ ...
        └── Environment 2/
            ├── Subject_1/ ...
            └── ...

주의: Subject_*.zip 형태로 되어 있으면 반드시 압축 해제 후 배치해야 합니다.

3. 환경 설치
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy scikit-learn matplotlib pandas tqdm

4. 학습 실행
python -m model.pretrained.train --epochs 30 --batch_size 32 --num_workers 0
- --num_workers 0: Windows 환경에서는 필수
- VRAM이 부족할 경우: --batch_size 16 또는 --batch_size 8로 낮추기

학습 결과물 위치
model/pretrained/checkpoints/
├── best.pt           # fall recall 기준 최고 모델
├── last.pt           # 마지막 epoch 모델
├── best_metrics.json
├── final_metrics.json
└── history.json