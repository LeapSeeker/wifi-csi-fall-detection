import numpy as np

for fname in ['model/finetune/cache/dummy_fall_peak.npz', 'model/finetune/cache/aug_fall_raw.npz']:
    d = np.load(fname, allow_pickle=True)
    X = d['X']
    y = d['y']
    print(f'=== {fname} ===')
    print(f'  X shape: {X.shape}')
    print(f'  fall={int((y==0).sum())}  non-fall={int((y!=0).sum())}')
    if 'is_augmented' in d:
        ia = d['is_augmented']
        print(f'  is_augmented: True={int(ia.sum())}  False={int((~ia).sum())}')
    if 'within_file_index' in d:
        wfi = d['within_file_index'][y==0]
        uniq = sorted(set(wfi.tolist()))
        print(f'  fall within_file_index: {uniq}')
    if 'augment_type' in d:
        atypes = sorted(set(d['augment_type'].tolist()))
        print(f'  augment_types: {atypes}')
    if 'shift_step' in d:
        print(f'  shift_step unique: {sorted(set(d["shift_step"].tolist()))}')
    print()
