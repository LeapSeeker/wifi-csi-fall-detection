"""각 클래스의 SDP 특성 분석: 낙상=spike+quiet, 걷기=periodic, 눕기=low 검증."""
import sys
import numpy as np

sys.path.insert(0, '.')
from model.finetune.train import SafeSignalDataset, split_safesignal_within_subject
import json

d_raw = np.load('model/finetune/cache/ss_peak160_p6.npz', allow_pickle=True)
split = json.loads(open('model/finetune/cache/split_seed42.json').read())
train_files = set(split['train'])

X         = d_raw['X']           # (N, 1, 28, 20)
y         = d_raw['y']
filenames = d_raw['filename']
activities= d_raw['activity']
wfi       = d_raw['within_file_index']

train_mask = np.array([fn in train_files for fn in filenames])

CLASS_NAMES = ['fall', 'walking', 'sit_stand', 'lying', 'standing', 'picking']

print('=== 클래스별 SDP 에너지 (L2 norm) 통계 ===')
print(f"{'클래스':12s}  {'mean':6s}  {'std':6s}  {'max':6s}")
for cls_id, cls_name in enumerate(CLASS_NAMES):
    mask = train_mask & (y == cls_id)
    if not mask.any():
        continue
    x_cls = X[mask]  # (N, 1, 28, 20)
    # 샘플별 전체 에너지
    energy = np.linalg.norm(x_cls.reshape(x_cls.shape[0], -1), axis=1)
    print(f'  {cls_name:12s}  {energy.mean():.2f}  {energy.std():.2f}  {energy.max():.2f}')

print()
print('=== fall 윈도우 시간축 에너지 프로파일 (wfi=0, 스텝별 평균) ===')
fall_mask = train_mask & (y == 0) & (wfi == 0)
x_fall = X[fall_mask]  # fall samples with wfi=0 (onset at ~step 9)
# 시간축(axis=2) 에너지: (N, 28)
t_energy_fall = np.linalg.norm(x_fall.reshape(x_fall.shape[0], 28, 20), axis=2)  # (N, 28)
mean_profile = t_energy_fall.mean(axis=0)  # (28,)
std_profile  = t_energy_fall.std(axis=0)

print('  step | mean  std')
for t in range(28):
    bar = '#' * int(mean_profile[t] * 3)
    print(f'   {t:2d}  | {mean_profile[t]:.3f}  {std_profile[t]:.3f}  {bar}')

print()
print('=== 비낙상 클래스 시간축 에너지 프로파일 ===')
for cls_id, cls_name in enumerate(['walking', 'lying', 'standing']):
    real_cls = {'walking': 1, 'lying': 3, 'standing': 4}[cls_name]
    mask = train_mask & (y == real_cls)
    if not mask.any():
        continue
    x_cls = X[mask]
    t_energy = np.linalg.norm(x_cls.reshape(x_cls.shape[0], 28, 20), axis=2)
    mean_p = t_energy.mean(axis=0)
    print(f'  [{cls_name}] mean energy per step: {mean_p.mean():.3f} ± {mean_p.std():.3f}')
    print(f'    t=0~9:   {mean_p[:10].mean():.3f}')
    print(f'    t=10~19: {mean_p[10:20].mean():.3f}')
    print(f'    t=20~27: {mean_p[20:].mean():.3f}')

print()
# spike 탐지: fall wfi=0에서 최대 에너지가 어느 스텝에 있는지
peak_steps = t_energy_fall.argmax(axis=1)
from collections import Counter
peak_dist = Counter(peak_steps.tolist())
print('=== fall wfi=0 에너지 피크 위치 분포 ===')
for s in sorted(peak_dist):
    print(f'  step {s:2d}: {peak_dist[s]}회')

# 각 phase 통계 추출 (for synthetic generation)
print()
print('=== Phase 통계 (synthetic 생성용) ===')
# pre-fall: steps 0~6
pre_fall = x_fall[:, 0, :7, :]   # (N, 7, 20)
spike    = x_fall[:, 0, 7:13, :] # (N, 6, 20) — 피크 구간
post_fall= x_fall[:, 0, 13:, :]  # (N, 15, 20)

# non-fall background
nf_mask = train_mask & (y != 0)
x_nf = X[nf_mask][:, 0, :, :]  # (N, 28, 20)

print(f'  pre-fall   mean={pre_fall.mean():.4f}  std={pre_fall.std():.4f}')
print(f'  spike      mean={spike.mean():.4f}  std={spike.std():.4f}')
print(f'  post-fall  mean={post_fall.mean():.4f}  std={post_fall.std():.4f}')
print(f'  non-fall   mean={x_nf.mean():.4f}  std={x_nf.std():.4f}')
print()
# subcarrier별로도 계산
pre_sc  = pre_fall.mean(axis=(0,1))  # (20,)
spk_sc  = spike.mean(axis=(0,1))
post_sc = post_fall.mean(axis=(0,1))
nf_sc   = x_nf.mean(axis=(0,1))
print(f'  pre_fall  by subcarrier: min={pre_sc.min():.3f} max={pre_sc.max():.3f}')
print(f'  spike     by subcarrier: min={spk_sc.min():.3f} max={spk_sc.max():.3f}')
print(f'  post_fall by subcarrier: min={post_sc.min():.3f} max={post_sc.max():.3f}')
print(f'  non_fall  by subcarrier: min={nf_sc.min():.3f} max={nf_sc.max():.3f}')
