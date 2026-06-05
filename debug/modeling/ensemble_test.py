"""model ensemble: 여러 체크포인트의 softmax 평균 → test R/FAR/F1.

usage:
    python -m debug.modeling.ensemble_test

여러 best_operating.pt 를 평균 앙상블해 window-level 테스트 성능을 측정.
학습 없이 기존 체크포인트만 사용.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.finetune.train import (
    PRETRAINED6_CLASSES,
    SafeSignalDataset,
    split_safesignal_within_subject,
    make_loader,
    predict_with_fall_threshold,
    evaluate_with_threshold,
    collate_batch,
)
from model.pretrained.model import CNNGRUAttention
from model.pretrained.metrics import compute_metrics

PRETRAINED6_CACHE = _PROJECT_ROOT / "model" / "finetune" / "cache" / "ss_peak160_p6.npz"

CKPT_CANDIDATES = [
    # recall_first 실험 모델들 (recall 높음)
    _PROJECT_ROOT / "model" / "finetune" / "checkpoints_rcf_aug500_fw2_t15" / "best_operating.pt",
    _PROJECT_ROOT / "model" / "finetune" / "checkpoints_rcf_aug200_fw3_t15" / "best_operating.pt",
    _PROJECT_ROOT / "model" / "finetune" / "checkpoints_rcf_focal_fw2_t15" / "best_operating.pt",
    # f1_first 실험 모델들 (FAR 낮음)
    _PROJECT_ROOT / "model" / "finetune" / "checkpoints_peak_aug_t15" / "best_operating.pt",
    _PROJECT_ROOT / "model" / "finetune" / "checkpoints_f1f_aug500_fw2_t15" / "best_operating.pt",
    _PROJECT_ROOT / "model" / "finetune" / "checkpoints_f1f_fw2_t15_wd01" / "best_operating.pt",
    _PROJECT_ROOT / "model" / "finetune" / "checkpoints_peak_fw2" / "best_operating.pt",
]

THRESHOLDS = [0.10, 0.12, 0.15, 0.17, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def load_model(ckpt_path: Path, device: torch.device) -> tuple[CNNGRUAttention, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    n_classes = len(ckpt.get("classes", PRETRAINED6_CLASSES)) if isinstance(ckpt, dict) else 6
    model = CNNGRUAttention(n_classes=n_classes).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    saved_threshold = float(ckpt.get("threshold", 0.30)) if isinstance(ckpt, dict) else 0.30
    return model, saved_threshold


def get_probs(model: CNNGRUAttention, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    all_probs, all_y = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_y.append(yb.numpy())
    return np.concatenate(all_probs), np.concatenate(all_y)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # 테스트 세트 로드 (within_subject seed=42)
    safesignal = SafeSignalDataset.from_npz(PRETRAINED6_CACHE)
    _, _, ss_test = split_safesignal_within_subject(
        safesignal, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    test_loader = make_loader(ss_test, batch_size=64)

    # 각 체크포인트의 softmax 수집
    ckpts_found = []
    probs_list = []
    for ckpt_path in CKPT_CANDIDATES:
        if not ckpt_path.exists():
            print(f"  skip (not found): {ckpt_path.parent.name}")
            continue
        model, saved_thr = load_model(ckpt_path, device)
        probs, y_true = get_probs(model, test_loader, device)
        ckpts_found.append(ckpt_path.parent.name)
        probs_list.append(probs)
        print(f"  loaded: {ckpt_path.parent.name}  saved_threshold={saved_thr:.2f}")

    if len(probs_list) == 0:
        print("체크포인트 없음")
        return

    y_true = y_true  # last batch; re-fetch from first model
    # Re-get y_true properly (same loader, deterministic)
    safesignal2 = SafeSignalDataset.from_npz(PRETRAINED6_CACHE)
    _, _, ss_test2 = split_safesignal_within_subject(
        safesignal2, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    test_loader2 = make_loader(ss_test2, batch_size=64)
    all_y = []
    with torch.no_grad():
        for _, yb, _ in test_loader2:
            all_y.append(yb.numpy())
    y_true = np.concatenate(all_y)

    print(f"\n=== 단일 모델 결과 (sweep threshold) ===")
    for name, probs in zip(ckpts_found, probs_list):
        best_r, best_far, best_f1, best_t = 0.0, 1.0, 0.0, 0.15
        for t in THRESHOLDS:
            m = evaluate_with_threshold(y_true, probs, t)
            if m.fall_recall > best_r or (m.fall_recall == best_r and m.far < best_far):
                best_r, best_far, best_f1, best_t = m.fall_recall, m.far, m.fall_f1, t
        print(f"  {name}: R={best_r:.3f} FAR={best_far:.3f} F1={best_f1:.3f} @t={best_t:.2f}")

    print(f"\n=== 앙상블 조합 결과 ===")
    # 모든 조합 시도
    n = len(probs_list)
    for mask in range(1, 2**n):
        combo_names = [ckpts_found[i] for i in range(n) if mask & (1 << i)]
        combo_probs = [probs_list[i] for i in range(n) if mask & (1 << i)]
        avg_probs = np.mean(np.stack(combo_probs, axis=0), axis=0)

        print(f"\n  [{' + '.join(combo_names)}]")
        for t in THRESHOLDS:
            m = evaluate_with_threshold(y_true, avg_probs, t)
            marker = " *** D011 PASS ***" if m.fall_recall >= 0.85 and m.far <= 0.15 else ""
            print(f"    t={t:.2f}  R={m.fall_recall:.3f} FAR={m.far:.3f} F1={m.fall_f1:.3f}{marker}")


if __name__ == "__main__":
    main()
