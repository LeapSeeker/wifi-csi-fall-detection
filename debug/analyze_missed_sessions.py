"""
test fall 세션 중 baseline이 놓치는 세션의 패턴을 분석한다.
- 활동 유형별 (FALL_SIT/FALL_STD/FALL_WALK × B/F) 미스율
- within_file_index별 미스율 (early/mid/late window position)
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from model.finetune.train import (
    SafeSignalDataset, split_safesignal_within_subject,
    make_loader, evaluate_with_threshold,
)
from model.pretrained.model import CNNGRUAttention

DEVICE = torch.device('cuda')
THRESHOLD = 0.15
FALL_CLS = 0

d_raw = np.load('model/finetune/cache/ss_peak160_p6.npz', allow_pickle=True)
ds = SafeSignalDataset.from_npz('model/finetune/cache/ss_peak160_p6.npz')
_, _, te = split_safesignal_within_subject(ds, val_ratio=0.2, test_ratio=0.2, seed=42)

test_idx = np.array(te.indices)
filenames  = d_raw['filename'][test_idx]
activities = d_raw['activity'][test_idx]
wfi        = d_raw['within_file_index'][test_idx]
y_true     = d_raw['y'][test_idx]

loader = make_loader(te, batch_size=64)

ckpt = torch.load('model/finetune/checkpoints_peak_aug_t15/best_operating.pt',
                  map_location=DEVICE, weights_only=False)
m = CNNGRUAttention(n_classes=6).to(DEVICE)
m.load_state_dict(ckpt['model'])
m.eval()
probs = []
with torch.no_grad():
    for x, _, _ in loader:
        probs.append(torch.softmax(m(x.to(DEVICE)), 1).cpu().numpy())
probs = np.concatenate(probs)

fall_prob  = probs[:, FALL_CLS]
pred_fall  = (fall_prob >= THRESHOLD).astype(int)

fall_mask  = y_true == FALL_CLS
print(f'Total test fall windows: {fall_mask.sum()}')
print(f'Correctly predicted: {int((pred_fall[fall_mask]).sum())}')
print()

# ── 활동 유형별 ────────────────────────────────────────────────────────
print('=== 활동 유형별 recall ===')
for act in sorted(set(activities[fall_mask].tolist())):
    m_act = fall_mask & (activities == act)
    tp = int(pred_fall[m_act].sum())
    total = int(m_act.sum())
    print(f'  {act:15s}: {tp}/{total} = {tp/total:.3f}')

print()

# ── within_file_index별 ────────────────────────────────────────────────
print('=== within_file_index(윈도우 위치)별 recall ===')
for w in sorted(set(wfi[fall_mask].tolist())):
    m_w = fall_mask & (wfi == w)
    tp = int(pred_fall[m_w].sum())
    total = int(m_w.sum())
    print(f'  wfi={w}: {tp}/{total} = {tp/total:.3f}')

print()

# ── 세션별 성공 수 (몇 개 윈도우가 올바른지) ──────────────────────────
print('=== 세션별 성공 윈도우 수 분포 ===')
fall_files = np.unique(filenames[fall_mask])
session_scores = {}
for fn in fall_files:
    m_fn = fall_mask & (filenames == fn)
    n_correct = int(pred_fall[m_fn].sum())
    n_total   = int(m_fn.sum())
    session_scores[fn] = (n_correct, n_total)

from collections import Counter
score_dist = Counter(v[0] for v in session_scores.values())
for k in sorted(score_dist):
    print(f'  {k}/{list(session_scores.values())[0][1]} correct: {score_dist[k]} sessions')

print()
print('=== 0/3 정답 (완전 미스) 세션 목록 ===')
for fn, (nc, nt) in sorted(session_scores.items()):
    if nc == 0:
        act = activities[filenames == fn][0]
        print(f'  {fn}  [{act}]')

print()
print('=== 1/3 정답 (부분 미스) 세션 목록 및 어떤 wfi가 틀렸는지 ===')
for fn, (nc, nt) in sorted(session_scores.items()):
    if nc == 1:
        act = activities[filenames == fn][0]
        m_fn = fall_mask & (filenames == fn)
        wfis_correct = wfi[m_fn & (pred_fall == 1)].tolist()
        wfis_wrong   = wfi[m_fn & (pred_fall == 0)].tolist()
        print(f'  {fn}  [{act}]  correct_wfi={wfis_correct} wrong_wfi={wfis_wrong}')
