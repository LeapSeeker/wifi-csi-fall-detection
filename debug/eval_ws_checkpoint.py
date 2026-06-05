"""cross_subject best_operating.pt를 within_subject test에 직접 평가 (0.912 출처 역추적)."""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")

from model.pretrained.model import CNNGRUAttention
from model.finetune.train import (
    split_safesignal_within_subject,
    predict_with_fall_threshold,
    evaluate_with_threshold,
)
from torch.utils.data import DataLoader

SS_CACHE = "model/finetune/cache/ss_peak160_p6.npz"
CKPT = "model/finetune/checkpoints_rcf_sdp_shift_sr90/best_operating.pt"
SEED = 42
N_CLASSES = 6

ckpt_data = torch.load(CKPT, map_location="cpu")
model = CNNGRUAttention(n_classes=N_CLASSES)
model.load_state_dict(ckpt_data["model"])
model.eval()

stored_thr = ckpt_data.get("threshold", None)
pv = ckpt_data.get("primary_val_metrics", {})
print("Stored threshold:", stored_thr)
print("Stored epoch:", ckpt_data.get("epoch", "N/A"))
print("Stored class_policy:", ckpt_data.get("class_policy", "N/A"))
print("Stored classes:", ckpt_data.get("classes", "N/A"))
print("Stored val_R:", pv.get("fall_recall", "N/A"))
print("Stored val_FAR:", pv.get("far", "N/A"))
print("Stored val_F1:", pv.get("fall_f1", "N/A"))

from model.finetune.train import SafeSignalDataset
ss_ds = SafeSignalDataset.from_npz(SS_CACHE)
train_s, val_s, test_s = split_safesignal_within_subject(
    ss_ds, seed=SEED, val_ratio=0.2, test_ratio=0.2
)
print(f"within_subject splits: train={len(train_s)} val={len(val_s)} test={len(test_s)}")
fall_test_count = sum(1 for i in test_s.indices if ss_ds.y[i] == 0)
print(f"  test fall samples: {fall_test_count}")

loader = DataLoader(test_s, batch_size=64, shuffle=False, num_workers=0)

all_probs_mat, all_y = [], []
with torch.no_grad():
    for batch in loader:
        X, y = batch[0], batch[1]
        logits = model(X.float())
        probs = torch.softmax(logits, dim=-1).numpy()
        all_probs_mat.extend(probs.tolist())
        all_y.extend(y.tolist())

y_true = np.array(all_y)
probs_arr = np.array(all_probs_mat)  # (N, C) full softmax

print("\n=== cross_subject best_operating.pt @ within_subject test ===")
for thr in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
    m = evaluate_with_threshold(y_true, probs_arr, float(thr))
    print(f"  t={thr:.2f}  R={m.fall_recall:.3f}  FAR={m.far:.3f}  F1={m.fall_f1:.3f}")

if stored_thr is not None:
    print(f"\n  stored threshold={stored_thr:.2f}:")
    m = evaluate_with_threshold(y_true, probs_arr, float(stored_thr))
    print(f"  R={m.fall_recall:.3f}  FAR={m.far:.3f}  F1={m.fall_f1:.3f}")
