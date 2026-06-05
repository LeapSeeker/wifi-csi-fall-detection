"""단일 CSV 파일로 즉시 낙상 판정.

사용:
    python infer_csv.py <CSV경로> [--threshold 0.70]

예시:
    python infer_csv.py dummy_src/E1/S01/E1_S01_A_FALL_SIT_B_T001.csv
    python infer_csv.py dummy_src/E1/S01/E1_S01_A_WALK_T001.csv
"""
import argparse, sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model.pretrained.model import CNNGRUAttention
from model.preprocessing.pipeline import preprocess_safesignal_file_full

CKPT      = Path("model/finetune/checkpoints_v6/best_operating.pt")
THRESHOLD = 0.70
CLASSES   = ["fall", "walking", "sit_stand", "lying", "standing", "picking"]

def run(csv_path: str, threshold: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = CNNGRUAttention(n_classes=6).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"처리 중: {csv_path}")
    result = preprocess_safesignal_file_full(csv_path, tail_window=False)
    X = torch.tensor(result.inputs, dtype=torch.float32).to(device)

    if X.shape[0] == 0:
        print("윈도우 없음")
        return

    with torch.no_grad():
        logits = model(X)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

    # 윈도우별 fall 확률
    fall_probs = probs[:, 0]
    max_fall   = float(fall_probs.max())
    predicted  = "FALL" if max_fall >= threshold else CLASSES[int(probs.mean(axis=0).argmax())]

    print(f"\n{'='*40}")
    print(f"  윈도우 수    : {X.shape[0]}")
    print(f"  최고 fall 확률: {max_fall:.3f}  (threshold={threshold})")
    print(f"  판정         : {'[FALL]' if predicted=='FALL' else '[----]'} {predicted}")
    print(f"{'='*40}")

    if X.shape[0] > 1:
        print("\n  [윈도우별 fall 확률]")
        for i, p in enumerate(fall_probs):
            bar = "#" * int(p * 20)
            mark = " <-- FALL" if p >= threshold else ""
            print(f"  win{i}: {p:.3f}  {bar}{mark}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="SafeSignal CSV 경로")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    run(args.csv, args.threshold)
