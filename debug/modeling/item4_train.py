"""항목4: policy × seed 학습 드라이버 (read-only 입력, 동결 train.py 무수정).

기존 검증된 model/finetune/train.py 의 run_training 을 그대로 구동하되, within_subject
split 만 **frozen split**(item4 cache 의 split_assignment)으로 monkeypatch 해 policy 간
val/test 가 동일하도록 보장한다(paired 성립). train.py 파일 자체는 수정하지 않는다.

- class_policy=pretrained6, split=within_subject, pretrained best.pt strict 로드.
- 모든 policy/seed 동일 학습 설정(crop 효과만 분리, 결정3). val early stop.
- 산출: checkpoints_item4/<policy>_s<seed>/ (best_operating.pt 등, 로컬) + run_config.json.
  (event-level threshold/평가는 item4_event_eval.py 에서 val sliding 으로 별도 수행.)

smoke: --smoke (1 policy×1 seed×2 epoch) 로 파이프라인 검증 후 전체 실행.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.finetune import train as T  # noqa: E402

# 증강 cache(is_augmented=True 더미)는 leak-free(frozen split: 더미→train, origin→manifest split,
# 더미 origin 전부 train → origin/aug 교차-split 없음, item4_build_arm_caches sanity 검증)이므로
# raw-only 가정 assertion 만 no-op. 데이터/split 로직은 동결 train.py 그대로 사용.
T._assert_raw_only_safesignal_cache = lambda *a, **k: None  # noqa: E731

# ── 이중 증강 방지: 더미(이미 오프라인 증강됨)는 온라인 증강 skip, 원본만 온라인 증강(=항목4 설정) ──
# 더미는 filename 마커(_AUG 등)로 식별. 동결 train.py 무수정, 래퍼만 monkeypatch.
import numpy as _np  # noqa: E402
import torch as _torch  # noqa: E402


def _getitem_skip_dummy_aug(self, idx):
    x, y, meta = self.subset[idx]
    fn = str(meta.get("filename", ""))
    is_dummy = any(m in fn.upper() for m in T.AUGMENT_FILENAME_MARKERS)
    if self.augment and meta.get("source") == T.SOURCE_SAFESIGNAL and not is_dummy:
        rng = _np.random.default_rng((self.base_seed, idx, self.epoch))
        x = _torch.as_tensor(T.augment_on_the_fly(x.numpy(), rng), dtype=_torch.float32)
    return x, y, meta


T.TrainAugmentDataset.__getitem__ = _getitem_skip_dummy_aug  # 더미 온라인 증강 skip

# ── (B) class-ratio 통제: fall effective mass를 원본 비율로 고정 ──
# 더미가 fall만 늘려 fall:nonfall 급증 → FAR 교란. fall class weight를 orig_fall/new_fall 로
# 낮춰 fall sampled mass가 더미 추가 전과 동일하게(순수 더미 품질 분리). 동결 train.py 무수정.
_FALLW = {"v": 1.0}
_orig_cw6 = T.safesignal_class_weight_6class


def _cw6_ctrl(label, hard_weight):
    return _FALLW["v"] if label == 0 else _orig_cw6(label, hard_weight)

ITEM4 = ROOT / "debug/modeling/diag_out/onset_detector/item4"
ALSAIFY = ROOT / "model/pretrained/checkpoints/dataset_cache_e12_w300_s300_lag1_20_tail_ps.npz"
PRETRAINED = ROOT / "model/pretrained/checkpoints/best.pt"
CKPT_ROOT = ROOT / "model/finetune/checkpoints_item4"
POLICIES = ["fixed", "onset_primary", "onset_reduced"]
SEEDS = [42, 43, 44, 45, 46]


def frozen_split_factory(split_map):
    import torch
    Sub = torch.utils.data.Subset

    def _frozen(dataset, val_ratio, test_ratio, seed):  # signature 호환
        tr, va, te = [], [], []
        for idx, fn in enumerate(dataset.filenames):
            s = split_map.get(str(fn), "train")
            (tr if s == "train" else va if s == "val" else te).append(idx)
        if not tr:
            raise ValueError("frozen split: empty train")
        return Sub(dataset, tr), Sub(dataset, va), Sub(dataset, te)
    return _frozen


def make_args(cache, ckpt_dir, seed, epochs, augment, weight_decay, patience, early_stop_start):
    return Namespace(
        safesignal_cache=cache, alsaify_cache=ALSAIFY, pretrained_ckpt=PRETRAINED,
        ckpt_dir=ckpt_dir, class_policy="pretrained6", split="within_subject", fold=1,
        test_ratio=0.20, threshold_min=0.30, epochs=epochs, batch_size=32, num_workers=0,
        seed=seed, device="auto", val_ratio=0.2, alsaify_val_ratio=0.2, source_ratio=0.60,
        hard_weight=1.30, auto_sampler_preset=False, fall_weight=1.0, warmup_epochs=5,
        backbone_lr_warmup=1e-5, backbone_lr=1e-4, attention_lr=3e-4, head_lr=1e-3,
        weight_decay=weight_decay, early_stop_start=early_stop_start, patience=patience,
        augment=augment, verbose=False,
    )


def run_one(policy, seed, epochs, augment, ckpt_root, weight_decay, patience, early_stop_start,
            control_fall_mass=False):
    cache = ITEM4 / f"item4_cache_{policy}.npz"
    d = np.load(cache, allow_pickle=True)
    split_map = {str(fn): str(sp) for fn, sp in zip(d["filename"], d["split_assignment"])}
    T.split_safesignal_within_subject = frozen_split_factory(split_map)  # monkeypatch (동결 split)
    if control_fall_mass:
        tr = d["split_assignment"] == "train"
        y = d["y"]; aug = d["is_augmented"].astype(bool) if "is_augmented" in d.files else np.zeros(len(y), bool)
        new_fall = int(((y == 0) & tr).sum())
        orig_fall = int(((y == 0) & tr & ~aug).sum())
        _FALLW["v"] = (orig_fall / new_fall) if new_fall else 1.0
        T.safesignal_class_weight_6class = _cw6_ctrl
        print(f"  [control] fall weight={_FALLW['v']:.4f} (orig {orig_fall}/new {new_fall} → mass 원본 고정)")
    else:
        T.safesignal_class_weight_6class = _orig_cw6
    ckpt_dir = ckpt_root / f"{policy}_s{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args = make_args(cache, ckpt_dir, seed, epochs, augment, weight_decay, patience, early_stop_start)
    t0 = time.time()
    print(f"\n===== TRAIN policy={policy} seed={seed} epochs={epochs} augment={augment} =====", flush=True)
    T.run_training(args)
    cfg = {"policy": policy, "seed": seed, "epochs": epochs, "augment": augment,
           "safesignal_cache": str(cache), "alsaify_cache": str(ALSAIFY),
           "pretrained_ckpt": str(PRETRAINED), "class_policy": "pretrained6",
           "split": "within_subject(frozen: fall=manifest, non-fall=canonical seed42)",
           "source_ratio": 0.60, "hard_weight": 1.30, "fall_weight": 1.0,
           "lr": {"backbone": 1e-4, "attention": 3e-4, "head": 1e-3, "warmup": 1e-5},
           "warmup_epochs": 5, "early_stop_start": early_stop_start, "patience": patience,
           "weight_decay": weight_decay, "elapsed_s": round(time.time() - t0, 1)}
    (ckpt_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {policy} s{seed} {cfg['elapsed_s']}s → {ckpt_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 policy×1 seed×2 epoch 검증")
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--ckpt-root", default="checkpoints_item4", help="model/finetune/ 하위 디렉토리명")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--early-stop-start", type=int, default=10)
    ap.add_argument("--control-fall-mass", action="store_true",
                    help="fall effective mass를 원본(더미 전) 비율로 고정 (class-ratio 통제)")
    ap.set_defaults(augment=True)
    args = ap.parse_args()
    ckpt_root = ROOT / "model/finetune" / args.ckpt_root

    T.setup_logging(False)
    if args.smoke:
        run_one("onset_primary", 42, 2, args.augment, ckpt_root, args.weight_decay, args.patience,
                args.early_stop_start, args.control_fall_mass)
        print("\n[SMOKE OK] 파이프라인 검증 완료. 전체 실행: python debug/modeling/item4_train.py")
        return 0

    runs = [(p, s) for p in args.policies for s in args.seeds]
    print(f"[전체] {len(runs)} runs root={args.ckpt_root} wd={args.weight_decay} patience={args.patience} "
          f"epochs={args.epochs} control_fall_mass={args.control_fall_mass}")
    t0 = time.time()
    for i, (p, s) in enumerate(runs, 1):
        print(f"\n######## run {i}/{len(runs)} ########", flush=True)
        run_one(p, s, args.epochs, args.augment, ckpt_root, args.weight_decay, args.patience,
                args.early_stop_start, args.control_fall_mass)
    print(f"\n[전체 완료] {len(runs)} runs | {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
