"""balanced 증강 학습 wrapper (read-only 입력, 동결 train.py 무수정).

item4_train.py 구조를 복사·확장. balanced cache(fall+non-fall 더미)로 학습하되 arm별 6-class
effective share 를 원본(A/B) 비율로 보존(class-ratio 통제). 동결 train.py 는 monkeypatch 만:
  ① _assert_raw_only_safesignal_cache no-op
  ② TrainAugmentDataset.__getitem__ : 더미(filename marker) 온라인 증강 skip + runtime counter
  ③ split_safesignal_within_subject : frozen split(cache split_assignment)
  ④ safesignal_class_weight_6class : arm별 target_share[c]/n_c (effective share = target 보존)

effective share 검증: n_c=augmented train safesignal class c count, raw_w[c]=target/n_c,
build_sample_weights 가 source 내부 정규화 → within-safesignal share = target. class별 abs
error<=1e-3 (초과 시 학습 전 fail-fast). n_c==0 도 fail-fast.

policies: fixed_balanced_aug / onset_primary_balanced_aug (balanced). fixed / onset_primary
(A/B 재학습용, original cw). cache = item4_cache_{policy}.npz.

A/B 재사용: checkpoints_item4_reg/{fixed,onset_primary}_s{seed} 존재+config 일치 시 재사용,
아니면 --retrain-ab 로 동일 config 재학습. 리포트 balanced_aug/ab_reuse_check.json|md.

제약(read-only): 원본 cache·manifest·동결 train.py 무수정. 신규 ckpt/리포트만.
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
import numpy as _np  # noqa: E402
import torch as _torch  # noqa: E402

# ① raw-only assertion no-op (더미 cache 는 leak-free frozen split, item4_build_balanced sanity)
T._assert_raw_only_safesignal_cache = lambda *a, **k: None  # noqa: E731

ITEM4 = ROOT / "debug/modeling/diag_out/onset_detector/item4"
ALSAIFY = ROOT / "model/pretrained/checkpoints/dataset_cache_e12_w300_s300_lag1_20_tail_ps.npz"
PRETRAINED = ROOT / "model/pretrained/checkpoints/best.pt"
REG_ROOT = ROOT / "model/finetune/checkpoints_item4_reg"          # A/B 재사용 후보
BAL_REPORT = ROOT / "debug/modeling/balanced_aug"
SEEDS = [42, 43, 44, 45, 46]
BALANCED_POLICIES = ["fixed_balanced_aug", "onset_primary_balanced_aug"]
CKPT_NAME = "best_operating.pt"

# arm별 보존 target (= A/B original-cw effective share, 스펙 §4). class index = PRETRAINED6 순서.
TARGETS = {
    "fixed":         [0.1157, 0.2030, 0.2233, 0.1858, 0.1183, 0.1538],  # = fixed_balanced_aug
    "onset_primary": [0.0627, 0.2152, 0.2367, 0.1969, 0.1254, 0.1630],  # = onset_primary_balanced_aug
}
SHARE_TOL = 1e-3

_orig_cw6 = T.safesignal_class_weight_6class
_BAL_W = {"v": None}  # 현재 arm 의 per-class weight list (balanced) 또는 None(baseline)

# ② counter + 더미 온라인 증강 skip
#   실측 가드 = dummy_aug_skipped(더미가 aug-eligible 분기 진입했으나 skip된 실제 횟수).
#   dummy_aug_applied 는 구조상 0 인 tripwire(더미가 aug 적용 분기로 새면 잡음). augment=True 면
#   marker_seen == dummy_aug_skipped 여야 함(모든 더미 접근이 skip 경로).
_CNT = {"marker_seen": 0, "dummy_aug_applied": 0, "dummy_aug_skipped": 0, "nondummy_aug_applied": 0}


def _getitem_skip_dummy_aug(self, idx):
    x, y, meta = self.subset[idx]
    fn = str(meta.get("filename", ""))
    is_dummy = any(m in fn.upper() for m in T.AUGMENT_FILENAME_MARKERS)
    if is_dummy:
        _CNT["marker_seen"] += 1
    if self.augment and meta.get("source") == T.SOURCE_SAFESIGNAL:
        if is_dummy:
            _CNT["dummy_aug_skipped"] += 1          # 더미는 skip (이중 증강 방지)
        else:
            rng = _np.random.default_rng((self.base_seed, idx, self.epoch))
            x = _torch.as_tensor(T.augment_on_the_fly(x.numpy(), rng), dtype=_torch.float32)
            _CNT["nondummy_aug_applied"] += 1
    return x, y, meta


T.TrainAugmentDataset.__getitem__ = _getitem_skip_dummy_aug


def _balanced_cw(label, hard_weight):
    return _BAL_W["v"][label] if _BAL_W["v"] is not None else _orig_cw6(label, hard_weight)


# ④ frozen split
def frozen_split_factory(split_map):
    import torch
    Sub = torch.utils.data.Subset

    def _frozen(dataset, val_ratio, test_ratio, seed):
        tr, va, te = [], [], []
        for idx, fn in enumerate(dataset.filenames):
            s = split_map.get(str(fn), "train")
            (tr if s == "train" else va if s == "val" else te).append(idx)
        if not tr:
            raise ValueError("frozen split: empty train")
        return Sub(dataset, tr), Sub(dataset, va), Sub(dataset, te)
    return _frozen


def _train_safesignal_class_counts(cache):
    """from_npz 의 side-fall keep + frozen train 기준 safesignal class별 count."""
    d = np.load(cache, allow_pickle=True)
    fns = [str(x) for x in d["filename"]]
    keep = np.array([not any(mk in Path(n).stem.upper() for mk in T.SIDE_FALL_MARKERS) for n in fns])
    sp = np.array([str(x) for x in d["split_assignment"]])
    y = d["y"].astype(int)
    tr = (sp == "train") & keep
    return {c: int(((y == c) & tr).sum()) for c in range(6)}, d, keep, tr


def effective_share(n_by_c, cw_list):
    raw = {c: cw_list[c] * n_by_c[c] for c in range(6)}
    tot = sum(raw.values())
    return {c: raw[c] / tot for c in range(6)}


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


def base_policy_of(policy):
    return policy[:-len("_balanced_aug")] if policy.endswith("_balanced_aug") else policy


def setup_arm(policy, cache):
    """frozen split + class_weight_fn 구성. balanced 면 effective-share 검증 dict 반환."""
    d = np.load(cache, allow_pickle=True)
    split_map = {str(fn): str(sp) for fn, sp in zip(d["filename"], d["split_assignment"])}
    T.split_safesignal_within_subject = frozen_split_factory(split_map)
    info = {"policy": policy, "balanced": policy.endswith("_balanced_aug")}
    if info["balanced"]:
        target = TARGETS[base_policy_of(policy)]
        n_by_c, _, _, _ = _train_safesignal_class_counts(cache)
        zero = [c for c in range(6) if n_by_c[c] == 0]
        if zero:
            raise SystemExit(f"[FAIL-FAST] balanced cw: n_c==0 classes {zero} ({cache.name})")
        cw_list = [target[c] / n_by_c[c] for c in range(6)]
        _BAL_W["v"] = cw_list
        T.safesignal_class_weight_6class = _balanced_cw
        eff = effective_share(n_by_c, cw_list)
        err = {c: abs(eff[c] - target[c]) for c in range(6)}
        maxerr = max(err.values())
        info.update({"target": target, "n_by_class": n_by_c,
                     "effective_share": {c: round(eff[c], 6) for c in range(6)},
                     "max_share_abs_error": maxerr})
        if maxerr > SHARE_TOL:
            print(f"[FAIL-FAST] effective share tol 초과 maxerr={maxerr:.5f} > {SHARE_TOL}")
            for c in range(6):
                print(f"  c{c} eff={eff[c]:.5f} target={target[c]:.5f} err={err[c]:.5f}")
            raise SystemExit(5)
        print(f"  [balanced] effective share OK maxerr={maxerr:.2e} | n_by_class={n_by_c}")
    else:
        _BAL_W["v"] = None
        T.safesignal_class_weight_6class = _orig_cw6
        print(f"  [baseline] original cw (A/B)")
    return info


def dummy_static_counts(cache):
    d = np.load(cache, allow_pickle=True)
    fns = [str(x) for x in d["filename"]]
    keep = np.array([not any(mk in Path(n).stem.upper() for mk in T.SIDE_FALL_MARKERS) for n in fns])
    sp = np.array([str(x) for x in d["split_assignment"]])
    tr = (sp == "train") & keep
    aug = d["is_augmented"].astype(bool) if "is_augmented" in d.files else np.zeros(len(fns), bool)
    dummy_total = int((tr & aug).sum())
    marker = np.array([any(m in Path(n).stem.upper() for m in T.AUGMENT_FILENAME_MARKERS) for n in fns])
    marker_matched = int((tr & marker).sum())
    return dummy_total, marker_matched


def run_one(policy, seed, epochs, augment, ckpt_root, weight_decay, patience, early_stop_start):
    cache = ITEM4 / f"item4_cache_{policy}.npz"
    if not cache.exists():
        raise SystemExit(f"[FAIL-FAST] cache 없음: {cache} (먼저 item4_build_balanced_caches.py)")
    info = setup_arm(policy, cache)
    dummy_total, marker_matched = dummy_static_counts(cache)
    assert marker_matched == dummy_total, \
        f"marker_matched({marker_matched}) != dummy_total({dummy_total})"
    print(f"  [counter-static] dummy_total={dummy_total} marker_matched={marker_matched} (일치 OK)")

    for k in _CNT:
        _CNT[k] = 0
    ckpt_dir = ckpt_root / f"{policy}_s{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args = make_args(cache, ckpt_dir, seed, epochs, augment, weight_decay, patience, early_stop_start)
    t0 = time.time()
    print(f"\n===== TRAIN policy={policy} seed={seed} epochs={epochs} balanced={info['balanced']} =====", flush=True)
    T.run_training(args)

    # ② runtime counter 검증 (marker 기반 skip 실측)
    assert _CNT["dummy_aug_applied"] == 0, f"tripwire: 더미 온라인 증강 적용됨! {_CNT['dummy_aug_applied']}"
    if augment:
        assert _CNT["dummy_aug_skipped"] == _CNT["marker_seen"], \
            f"더미 skip 누락: dummy_aug_skipped {_CNT['dummy_aug_skipped']} != marker_seen {_CNT['marker_seen']}"
    print(f"  [counter-runtime] marker 기반 skip 검증: marker_seen={_CNT['marker_seen']} "
          f"dummy_aug_skipped={_CNT['dummy_aug_skipped']}(=marker_seen 기대) "
          f"dummy_aug_applied={_CNT['dummy_aug_applied']}(tripwire 0) "
          f"nondummy_aug_applied={_CNT['nondummy_aug_applied']}")
    cfg = {"policy": policy, "seed": seed, "epochs": epochs, "augment": augment, "balanced": info["balanced"],
           "safesignal_cache": str(cache), "class_policy": "pretrained6",
           "split": "within_subject(frozen)", "source_ratio": 0.60, "hard_weight": 1.30,
           "lr": {"backbone": 1e-4, "attention": 3e-4, "head": 1e-3, "warmup": 1e-5},
           "warmup_epochs": 5, "early_stop_start": early_stop_start, "patience": patience,
           "weight_decay": weight_decay,
           "balanced_info": {k: info[k] for k in info if k != "policy"},
           "counters": dict(_CNT), "dummy_total": dummy_total, "marker_matched": marker_matched,
           "elapsed_s": round(time.time() - t0, 1)}
    (ckpt_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {policy} s{seed} {cfg['elapsed_s']}s → {ckpt_dir}", flush=True)


def ab_reuse_check(seeds, epochs, weight_decay, patience, early_stop_start):
    """A/B(checkpoints_item4_reg) 재사용 가능성 점검 → 리포트."""
    BAL_REPORT.mkdir(parents=True, exist_ok=True)
    want = {"epochs": epochs, "weight_decay": weight_decay, "patience": patience,
            "early_stop_start": early_stop_start}
    res = {"reg_root": str(REG_ROOT), "want_config": want, "per_arm_seed": {}, "all_reusable": True}
    for arm in ("fixed", "onset_primary"):
        for seed in seeds:
            d = REG_ROOT / f"{arm}_s{seed}"
            ck = d / CKPT_NAME
            rc = d / "run_config.json"
            entry = {"ckpt_exists": ck.exists(), "config_exists": rc.exists(),
                     "config_match": False, "reusable": False, "diff": {}}
            if rc.exists():
                try:
                    cfg = json.loads(rc.read_text(encoding="utf-8"))
                    diff = {k: [want[k], cfg.get(k)] for k in want if cfg.get(k) != want[k]}
                    entry["diff"] = diff
                    entry["config_match"] = (len(diff) == 0)
                except Exception as e:
                    entry["diff"] = {"error": str(e)}
            entry["reusable"] = entry["ckpt_exists"] and entry["config_match"]
            if not entry["reusable"]:
                res["all_reusable"] = False
            res["per_arm_seed"][f"{arm}_s{seed}"] = entry
    (BAL_REPORT / "ab_reuse_check.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    L = ["# A/B(fixed/onset_primary) 재사용 점검\n",
         f"reg_root: {REG_ROOT}", f"want config: {want}", f"\nall_reusable: **{res['all_reusable']}**\n",
         "arm_seed | ckpt | config | match | reusable | diff", "---|---|---|---|---|---"]
    for k, e in res["per_arm_seed"].items():
        L.append(f"{k} | {e['ckpt_exists']} | {e['config_exists']} | {e['config_match']} | "
                 f"{e['reusable']} | {e['diff'] or '-'}")
    if not res["all_reusable"]:
        L.append("\n⚠ 재사용 불가 항목 존재 → `--retrain-ab` 로 A/B 동일 config 재학습 필요"
                 "(ckpt-root 에 fixed/onset_primary 생성).")
    (BAL_REPORT / "ab_reuse_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[A/B reuse] all_reusable={res['all_reusable']} → {BAL_REPORT}/ab_reuse_check.md")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--policies", nargs="+", default=BALANCED_POLICIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--ckpt-root", default="checkpoints_item4_balanced")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--early-stop-start", type=int, default=8)
    ap.add_argument("--check-ab-reuse", action="store_true", help="A/B 재사용 점검만 수행")
    ap.add_argument("--retrain-ab", action="store_true",
                    help="A/B(fixed,onset_primary) 를 동일 config 로 ckpt-root 에 재학습")
    ap.set_defaults(augment=True)
    args = ap.parse_args()
    ckpt_root = ROOT / "model/finetune" / args.ckpt_root
    T.setup_logging(False)

    if args.check_ab_reuse:
        ab_reuse_check(args.seeds, args.epochs, args.weight_decay, args.patience, args.early_stop_start)
        return 0

    if args.smoke:
        # preflight + A/B reuse + 4-arm effective share + 1 balanced policy 1 epoch dry run
        print("[smoke] balanced cache 존재 확인")
        for p in BALANCED_POLICIES:
            c = ITEM4 / f"item4_cache_{p}.npz"
            print(f"  {'OK' if c.exists() else 'MISSING'}  {c}")
            if not c.exists():
                raise SystemExit(f"[FAIL-FAST] {c} 없음 — 먼저 item4_build_balanced_caches.py")
        ab_reuse_check(args.seeds, args.epochs, args.weight_decay, args.patience, args.early_stop_start)
        print("[smoke] 4-arm effective share 검증 (A/B baseline cw, C/D balanced cw)")
        for p in ("fixed", "onset_primary"):  # A/B: original cw 가 target 재현하는지
            n_by_c, _, _, _ = _train_safesignal_class_counts(ITEM4 / f"item4_cache_{p}.npz")
            eff = effective_share(n_by_c, [_orig_cw6(c, 1.30) for c in range(6)])
            mx = max(abs(eff[c] - TARGETS[p][c]) for c in range(6))
            print(f"  A/B {p:14s} maxerr={mx:.2e} {'OK' if mx <= SHARE_TOL else 'FAIL'}")
            if mx > SHARE_TOL:
                raise SystemExit(5)
        for p in BALANCED_POLICIES:  # C/D: balanced cw
            setup_arm(p, ITEM4 / f"item4_cache_{p}.npz")  # 내부에서 tol assert
        print("[smoke] C dry run (fixed_balanced_aug, seed42, 1 epoch)")
        run_one("fixed_balanced_aug", 42, 1, args.augment, ckpt_root,
                args.weight_decay, args.patience, args.early_stop_start)
        print("\n[SMOKE OK] 본학습: python debug/modeling/item4_train_balanced.py "
              "--policies fixed_balanced_aug onset_primary_balanced_aug --seeds 42 43 44 45 46 "
              "--epochs 40 --weight-decay 1e-4 --patience 8 --early-stop-start 8 "
              "--ckpt-root checkpoints_item4_balanced")
        return 0

    policies = ["fixed", "onset_primary"] if args.retrain_ab else args.policies
    runs = [(p, s) for p in policies for s in args.seeds]
    print(f"[전체] {len(runs)} runs root={args.ckpt_root} policies={policies} "
          f"epochs={args.epochs} wd={args.weight_decay} patience={args.patience}")
    t0 = time.time()
    for i, (p, s) in enumerate(runs, 1):
        print(f"\n######## run {i}/{len(runs)} ########", flush=True)
        run_one(p, s, args.epochs, args.augment, ckpt_root, args.weight_decay,
                args.patience, args.early_stop_start)
    print(f"\n[전체 완료] {len(runs)} runs | {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
