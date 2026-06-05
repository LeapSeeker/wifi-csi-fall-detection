"""기존 aug_fall_raw.npz + 새 aug_fall_sdp_shifted.npz → aug_fall_merged.npz"""
import numpy as np
from pathlib import Path

files = [
    'model/finetune/cache/aug_fall_raw.npz',
    'model/finetune/cache/aug_fall_sdp_shifted.npz',
]

arrays = {k: [] for k in ['X', 'y', 'filename', 'subject', 'env', 'activity',
                           'trial', 'within_file_index', 'is_augmented', 'source']}

for f in files:
    d = np.load(f, allow_pickle=True)
    n = d['X'].shape[0]
    arrays['X'].append(d['X'])
    arrays['y'].append(d['y'])
    arrays['filename'].append(d['filename'])
    arrays['subject'].append(d['subject'])
    arrays['env'].append(d['env'])
    arrays['activity'].append(d['activity'])
    arrays['trial'].append(d['trial'])
    arrays['within_file_index'].append(d['within_file_index'])
    arrays['is_augmented'].append(d['is_augmented'])
    src_key = 'source' if 'source' in d else None
    if src_key:
        arrays['source'].append(d[src_key])
    else:
        arrays['source'].append(np.array(['unknown'] * n, dtype=object))
    print(f'{f}: {n} samples')

out = {}
for k, vlist in arrays.items():
    out[k] = np.concatenate(vlist)

print(f'\nTotal: {out["X"].shape[0]} samples')
print(f'Fall: {int((out["y"]==0).sum())}')

classes = np.load(files[0], allow_pickle=True)['classes']
np.savez('model/finetune/cache/aug_fall_merged.npz', classes=classes, **out)
print('저장: model/finetune/cache/aug_fall_merged.npz')
