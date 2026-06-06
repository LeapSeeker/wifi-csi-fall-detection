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


def make_args(cache, ckpt_dir, seed, epochs, augment):
    return Namespace(
        safesignal_cache=cache, alsaify_cache=ALSAIFY, pretrained_ckpt=PRETRAINED,
        ckpt_dir=ckpt_dir, class_policy="pretrained6", split="within_subject", fold=1,
        test_ratio=0.20, threshold_min=0.30, epochs=epochs, batch_size=32, num_workers=0,
        seed=seed, device="auto", val_ratio=0.2, alsaify_val_ratio=0.2, source_ratio=0.60,
        hard_weight=1.30, auto_sampler_preset=False, fall_weight=1.0, warmup_epochs=5,
        backbone_lr_warmup=1e-5, backbone_lr=1e-4, attention_lr=3e-4, head_lr=1e-3,
        weight_decay=0.0, early_stop_start=10, patience=12, augment=augment, verbose=False,
    )


def run_one(policy, seed, epochs, augment):
    cache = ITEM4 / f"item4_cache_{policy}.npz"
    d = np.load(cache, allow_pickle=True)
    split_map = {str(fn): str(sp) for fn, sp in zip(d["filename"], d["split_assignment"])}
    T.split_safesignal_within_subject = frozen_split_factory(split_map)  # monkeypatch (동결 split)
    ckpt_dir = CKPT_ROOT / f"{policy}_s{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args = make_args(cache, ckpt_dir, seed, epochs, augment)
    t0 = time.time()
    print(f"\n===== TRAIN policy={policy} seed={seed} epochs={epochs} augment={augment} =====", flush=True)
    T.run_training(args)
    cfg = {"policy": policy, "seed": seed, "epochs": epochs, "augment": augment,
           "safesignal_cache": str(cache), "alsaify_cache": str(ALSAIFY),
           "pretrained_ckpt": str(PRETRAINED), "class_policy": "pretrained6",
           "split": "within_subject(frozen: fall=manifest, non-fall=canonical seed42)",
           "source_ratio": 0.60, "hard_weight": 1.30, "fall_weight": 1.0,
           "lr": {"backbone": 1e-4, "attention": 3e-4, "head": 1e-3, "warmup": 1e-5},
           "warmup_epochs": 5, "early_stop_start": 10, "patience": 12,
           "elapsed_s": round(time.time() - t0, 1)}
    (ckpt_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {policy} s{seed} {cfg['elapsed_s']}s → {ckpt_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 policy×1 seed×2 epoch 검증")
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.set_defaults(augment=True)
    args = ap.parse_args()

    T.setup_logging(False)
    if args.smoke:
        run_one("onset_primary", 42, epochs=2, augment=args.augment)
        print("\n[SMOKE OK] 파이프라인 검증 완료. 전체 실행: python debug/modeling/item4_train.py")
        return 0

    runs = [(p, s) for p in args.policies for s in args.seeds]
    print(f"[전체] {len(runs)} runs: policies={args.policies} seeds={args.seeds} epochs={args.epochs}")
    t0 = time.time()
    for i, (p, s) in enumerate(runs, 1):
        print(f"\n######## run {i}/{len(runs)} ########", flush=True)
        run_one(p, s, args.epochs, args.augment)
    print(f"\n[전체 완료] {len(runs)} runs | {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
