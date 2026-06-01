"""SafeSignal fine-tuning / combined-training entry point.

Confirmed policy:
  - 7 classes: fall/walking/sit_stand/lying/standing/running/picking
  - Combined training from preconverted tensors, single DataLoader
  - Full unfreeze with 5-epoch backbone LR warmup
  - SafeSignal-only primary validation and sealed SafeSignal test folds

Implementation notes:
  This file intentionally keeps dataset conversion and thresholded evaluation as
  TODO seams. The train loop, sampler policy, checkpoint policy, and model-head
  migration are defined here so the missing pieces can be plugged in without
  changing the training contract.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.augment.augment import jittering, noise_scale, scaling, time_warping
from model.pretrained.metrics import FallMetrics, compute_metrics
from model.pretrained.model import CNNGRUAttention

LOG = logging.getLogger("safesignal.finetune")

FINETUNE_CLASSES: tuple[str, ...] = (
    "fall",
    "walking",
    "sit_stand",
    "lying",
    "standing",
    "running",
    "picking",
)
N_CLASSES = len(FINETUNE_CLASSES)

SOURCE_ALSAIFY = "alsaify"
SOURCE_SAFESIGNAL = "safesignal"
SourceName = Literal["alsaify", "safesignal"]

SIDE_FALL_MARKERS = ("FALL_SIT_S", "FALL_STD_S", "FALL_WALK_S")

FOLDS: dict[int, dict[str, list[int]]] = {
    1: {"train_subjects": [2, 3], "test_subjects": [1]},
    2: {"train_subjects": [1, 3], "test_subjects": [2]},
    3: {"train_subjects": [1, 2], "test_subjects": [3]},
}

OPERATING_RECALL_TARGET = 0.90
OPERATING_FAR_TARGET = 0.10
OPERATING_F1_TARGET = 0.85
STRETCH_RECALL_TARGET = 0.93
STRETCH_FAR_TARGET = 0.07


@dataclass(frozen=True)
class TensorRecord:
    x: torch.Tensor
    y: int
    subject: int
    source: SourceName
    env: int | None
    filename: str


def _to_list(values: object) -> list:
    """Normalize numpy/torch/iterable → plain list without breaking on strings."""
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, torch.Tensor):
        return values.tolist()
    if hasattr(values, "tolist") and not isinstance(values, (str, bytes)):
        return values.tolist()
    return list(values)


def _parse_subject(value: object) -> int:
    """Cache compat: accept int, np.integer, float, bytes, or "Snn"/"nn" strings."""
    if isinstance(value, bool):
        raise ValueError(f"subject must not be bool, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        iv = int(value)
        if iv != value:
            raise ValueError(f"non-integer float subject: {value!r}")
        return iv
    if isinstance(value, bytes):
        return _parse_subject(value.decode())
    if isinstance(value, str):
        s = value.strip().upper()
        if s.startswith("S"):
            s = s[1:]
        if not s.isdigit():
            raise ValueError(f"unparseable subject string: {value!r}")
        return int(s)
    raise ValueError(f"unrecognized subject value type: {type(value).__name__} ({value!r})")


def _parse_env(value: object) -> int | None:
    """Cache compat: accept int / "Enn" string / None."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return int(value)
    if isinstance(value, bytes):
        return _parse_env(value.decode())
    if isinstance(value, str):
        s = value.strip().upper()
        if not s or s in {"NONE", "NULL", "NA", "NAN"}:
            return None
        if s.startswith("E"):
            s = s[1:]
        if not s.isdigit():
            raise ValueError(f"unparseable env string: {value!r}")
        return int(s)
    raise ValueError(f"unrecognized env value type: {type(value).__name__} ({value!r})")


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    metrics: FallMetrics
    rule: str
    exceeded_far: bool = False


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    primary_val_loss: float
    primary_val: FallMetrics
    auxiliary_val_loss: float | None
    auxiliary_val: FallMetrics | None
    threshold: ThresholdResult


@dataclass(frozen=True)
class FoldReport:
    fold: int
    threshold: float
    metrics: FallMetrics
    flags: list[str]
    false_positive_by_class: dict[str, float]


class TensorMetadataDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict]]):
    """Dataset backed by preconverted tensors and side metadata."""

    def __init__(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        subjects: np.ndarray | torch.Tensor,
        sources: Iterable[str],
        envs: Iterable[int | None],
        filenames: Iterable[str],
    ) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.subjects = [_parse_subject(s) for s in _to_list(subjects)]
        self.sources = [str(s).lower() for s in sources]
        self.envs = [_parse_env(e) for e in _to_list(envs)]
        self.filenames = [str(f) for f in filenames]

        n = len(self.y)
        lengths = {
            "X": len(self.X),
            "subjects": len(self.subjects),
            "sources": len(self.sources),
            "envs": len(self.envs),
            "filenames": len(self.filenames),
        }
        bad = {k: v for k, v in lengths.items() if v != n}
        if bad:
            raise ValueError(f"metadata length mismatch: y={n}, bad={bad}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        meta = {
            "subject": self.subjects[idx],
            "source": self.sources[idx],
            "env": self.envs[idx],
            "filename": self.filenames[idx],
        }
        return self.X[idx], self.y[idx], meta


AUGMENT_FILENAME_MARKERS = ("_AUG", "AUGMENT", "JITTER", "SCALING", "TIMEWARP", "NOISE")

_AUGMENT_CACHE_HELP = (
    "SafeSignal augmentation must be applied after subject split via "
    "TrainAugmentDataset, not baked into the cache. The official train/val/test "
    "split is row-based, so mixing originals and their augmented copies into one "
    "cache leaks augmented samples across train/val/test."
)


def _assert_raw_only_safesignal_cache(data, filenames: list[str]) -> None:
    """Reject SafeSignal caches that already contain augmented samples.

    The default safe input is a raw-only cache. Augmented-cache support would
    require an origin-group split, which is intentionally out of scope here.

    TODO: augmented cache support requires origin-group split. ``origin_filename``
    / ``origin_session_id`` keys are allowed as raw provenance for now (not an
    error), but until the split groups by origin they cannot prevent
    original/augmented leakage, so augmented caches stay rejected.
    """
    keys = set(getattr(data, "files", []) or [])

    if "is_augmented" in keys:
        flags = np.asarray(data["is_augmented"])
        if flags.astype(bool).any():
            raise ValueError(
                "SafeSignal cache marks is_augmented=True for "
                f"{int(flags.astype(bool).sum())} sample(s). " + _AUGMENT_CACHE_HELP
            )

    if "augment_type" in keys:
        types = [str(x).strip() for x in np.asarray(data["augment_type"]).tolist()]
        nonempty = [x for x in types if x and x.lower() not in {"none", "raw", "original"}]
        if nonempty:
            raise ValueError(
                "SafeSignal cache carries non-empty augment_type values "
                f"(e.g. {sorted(set(nonempty))[:3]}). " + _AUGMENT_CACHE_HELP
            )

    flagged = [
        name for name in filenames
        if any(marker in name.upper() for marker in AUGMENT_FILENAME_MARKERS)
    ]
    if flagged:
        raise ValueError(
            "SafeSignal cache filenames look augmented "
            f"(e.g. {flagged[:3]}). " + _AUGMENT_CACHE_HELP
        )


class SafeSignalDataset(TensorMetadataDataset):
    """SafeSignal tensor dataset.

    TODO: SafeSignalDataset: CSV -> window tensor conversion and train-only
    augmentation will be implemented in a separate task. This constructor
    currently expects a prepared .npz with X/y/subject/source/env/filename.
    """

    @classmethod
    def from_npz(cls, path: Path) -> "SafeSignalDataset":
        data = np.load(path, allow_pickle=True)
        filenames = [str(x) for x in data["filename"].tolist()]
        _assert_raw_only_safesignal_cache(data, filenames)
        excluded = sum(
            any(marker in Path(name).stem.upper() for marker in SIDE_FALL_MARKERS)
            for name in filenames
        )
        if excluded:
            LOG.info("filtered side-fall SafeSignal samples: %d", excluded)

        keep = np.array(
            [
                not any(marker in Path(name).stem.upper() for marker in SIDE_FALL_MARKERS)
                for name in filenames
            ],
            dtype=bool,
        )
        sources = [SOURCE_SAFESIGNAL] * int(keep.sum())
        return cls(
            X=data["X"][keep],
            y=data["y"][keep],
            subjects=data["subject"][keep],
            sources=sources,
            envs=data["env"][keep].tolist(),
            filenames=np.asarray(filenames, dtype=object)[keep].tolist(),
        )


class AlsaifyDataset(TensorMetadataDataset):
    """Alsaify tensor dataset.

    TODO: AlsaifyDataset: reuse the existing model/pretrained/train.py
    preprocessing path in a separate task. This constructor currently expects
    a prepared .npz with X/y/subject/env/filename.
    """

    @classmethod
    def from_npz(cls, path: Path) -> "AlsaifyDataset":
        data = np.load(path, allow_pickle=True)
        n = int(len(data["y"]))
        sources = [SOURCE_ALSAIFY] * n
        y = data["y"].astype(np.int64, copy=True)
        # Alsaify pretrained cache is 6-class:
        #   0 fall, 1 walking, 2 sit_stand, 3 lying, 4 standing, 5 picking.
        # Fine-tuning is 7-class and inserts running at index 5, so picking
        # must move to index 6. Alsaify has no running samples.
        y[y == 5] = 6
        return cls(
            X=data["X"],
            y=y,
            subjects=data["subject"],
            sources=sources,
            envs=data["env"].tolist() if "env" in data else [None] * n,
            filenames=data["filename"].tolist() if "filename" in data else [""] * n,
        )

_ON_THE_FLY_AUGMENTS = (jittering, scaling, time_warping, noise_scale)


def augment_on_the_fly(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pick one of the four augmentations uniformly and apply it once.

    Reuses model/augment/augment.py functions with their default parameters
    (only ``rng`` is forwarded). Input/output shape (1, 28, 20), dtype float32.
    """
    fn = _ON_THE_FLY_AUGMENTS[int(rng.integers(len(_ON_THE_FLY_AUGMENTS)))]
    out = fn(x, rng=rng)
    return np.asarray(out, dtype=np.float32)


class TrainAugmentDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict]]):
    """Train-only augmentation wrapper.

    This wrapper exists to enforce the [D-019] augmentation scope contract:
    augmentation MUST be applied only to the train subset after the subject
    fold split — never to val/test/threshold selection. Wrapping a Subset here
    (rather than mutating the global SafeSignalDataset) makes that guarantee
    visible at the call site.

    On-the-fly augmentation (b안): when ``augment=True`` and a sample's source is
    SafeSignal, one of jittering/scaling/time_warping/noise_scale is drawn
    uniformly and applied once. The draw is a deterministic function of
    ``(base_seed, idx, epoch)`` so each epoch produces fresh-but-reproducible
    variants. Alsaify samples always pass through untouched. The original
    tensors in the backing dataset are never mutated — only new arrays are made.
    """

    def __init__(self, subset: Dataset, augment: bool = False, base_seed: int = 42) -> None:
        self.subset = subset
        self.augment = augment
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.subset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        x, y, meta = self.subset[idx]
        if self.augment and meta.get("source") == SOURCE_SAFESIGNAL:
            rng = np.random.default_rng((self.base_seed, idx, self.epoch))
            x_np = x.numpy()
            aug = augment_on_the_fly(x_np, rng)
            x = torch.as_tensor(aug, dtype=torch.float32)
        return x, y, meta


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor, dict]]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    xs, ys, metas = zip(*batch)
    merged = {
        "subject": [m["subject"] for m in metas],
        "source": [m["source"] for m in metas],
        "env": [m["env"] for m in metas],
        "filename": [m["filename"] for m in metas],
    }
    return torch.stack(list(xs), dim=0), torch.stack(list(ys), dim=0), merged


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_sampler_preset(n_safesignal: int) -> tuple[float, float]:
    """Return (SafeSignal source ratio, hard class weight)."""
    if n_safesignal < 2_000:
        return 0.50, 1.15
    if n_safesignal < 3_600:
        return 0.55, 1.20
    if n_safesignal < 5_500:
        return 0.55, 1.30
    return 0.60, 1.30


def safesignal_class_weight(label: int, hard_weight: float) -> float:
    name = FINETUNE_CLASSES[label]
    if name in {"picking", "sit_stand", "running"}:
        return hard_weight
    if name == "walking":
        return hard_weight * 0.88
    if name == "lying":
        return 1.05
    return 1.0


def build_sample_weights(
    dataset: Dataset,
    safesignal_ratio: float,
    hard_weight: float,
) -> torch.DoubleTensor:
    """Build sample weights that fix the source ratio and shape class mix.

    Two roles are kept independent:
      - ``safesignal_ratio`` fixes total sampling mass per source. After this
        function returns, ``weights[source==safesignal].sum() == safesignal_ratio``
        and ``weights[source==alsaify].sum() == 1 - safesignal_ratio``,
        regardless of class weights.
      - ``hard_weight`` only reshapes the *within-source* class distribution.

    Formula:
      raw_class_w        = class weight for the sample's (source, class)
      source_raw_sum[s]  = sum of raw_class_w over all samples of source s
      sample_weight      = source_quota[s] * raw_class_w / source_raw_sum[s]

    Because raw_class_w sums to source_raw_sum[s] within a source, the per-source
    mass collapses to source_quota[s] exactly — class weights only redistribute
    mass *inside* a source, never across sources.
    """
    if not 0.0 < safesignal_ratio < 1.0:
        raise ValueError(f"safesignal_ratio must be in (0, 1), got {safesignal_ratio}")

    records: list[tuple[int, str]] = []
    for i in range(len(dataset)):
        _, y, meta = dataset[i]
        records.append((int(y.item()), str(meta["source"]).lower()))

    n_safe = sum(1 for _, src in records if src == SOURCE_SAFESIGNAL)
    n_alsaify = sum(1 for _, src in records if src == SOURCE_ALSAIFY)
    if n_safe == 0 or n_alsaify == 0:
        raise ValueError(
            f"combined sampler requires both sources, got safesignal={n_safe}, "
            f"alsaify={n_alsaify}"
        )

    source_quota = {
        SOURCE_SAFESIGNAL: safesignal_ratio,
        SOURCE_ALSAIFY: 1.0 - safesignal_ratio,
    }

    # Pass 1: raw class weight per sample + per-source raw sum.
    raw_class_w: list[float] = []
    source_raw_sum: dict[str, float] = {SOURCE_SAFESIGNAL: 0.0, SOURCE_ALSAIFY: 0.0}
    for label, source in records:
        if source == SOURCE_SAFESIGNAL:
            cw = safesignal_class_weight(label, hard_weight)
        elif source == SOURCE_ALSAIFY:
            cw = 1.0
        else:
            raise ValueError(f"unknown source: {source!r}")
        raw_class_w.append(cw)
        source_raw_sum[source] += cw

    for source, raw_sum in source_raw_sum.items():
        if raw_sum <= 0.0:
            raise ValueError(f"non-positive raw class weight sum for source {source!r}")

    # Pass 2: normalize within source so each source's mass == its quota.
    weights = [
        source_quota[source] * cw / source_raw_sum[source]
        for (label, source), cw in zip(records, raw_class_w)
    ]
    return torch.as_tensor(weights, dtype=torch.double)


def split_safesignal_by_fold(
    dataset: SafeSignalDataset,
    fold: int,
    val_ratio: float,
    seed: int,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """Return train/val/test SafeSignal subsets by subject-sealed fold.

    E4 samples are separated strictly by subject because splitting keys use the
    subject metadata only.
    """
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {sorted(FOLDS)}, got {fold}")

    train_subjects = set(FOLDS[fold]["train_subjects"])
    test_subjects = set(FOLDS[fold]["test_subjects"])
    train_pool, test_idx = [], []
    for idx, subject in enumerate(dataset.subjects):
        if subject in test_subjects:
            test_idx.append(idx)
        elif subject in train_subjects:
            train_pool.append(idx)

    rng = np.random.default_rng(seed + fold)
    train_pool = np.asarray(train_pool, dtype=np.int64)
    rng.shuffle(train_pool)
    n_val = max(1, int(round(len(train_pool) * val_ratio))) if len(train_pool) else 0
    val_idx = train_pool[:n_val].tolist()
    train_idx = train_pool[n_val:].tolist()

    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx),
    )


def split_alsaify_subject_holdout(
    dataset: AlsaifyDataset,
    val_ratio: float,
    seed: int,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    subjects = sorted(set(dataset.subjects))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_ratio))) if len(subjects) > 1 else 0
    val_subjects = set(subjects[:n_val])
    train_idx, val_idx = [], []
    for idx, subject in enumerate(dataset.subjects):
        (val_idx if subject in val_subjects else train_idx).append(idx)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)


def migrate_pretrained_head(
    model: CNNGRUAttention,
    pretrained_path: Path,
    device: torch.device,
) -> None:
    """Load 6-class checkpoint and migrate classifier to 7 classes.

    Index policy:
      new[0:5] = old[0:5]  # fall..standing
      new[5]   = kaiming init  # running
      new[6]   = old[5]    # picking
    """
    ckpt = torch.load(pretrained_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    old_w_key = "classifier.1.weight"
    old_b_key = "classifier.1.bias"
    old_w = state.get(old_w_key)
    old_b = state.get(old_b_key)
    if old_w is None or old_b is None:
        raise KeyError("pretrained checkpoint missing classifier.1 weight/bias")
    if tuple(old_w.shape)[0] != 6 or tuple(old_b.shape)[0] != 6:
        raise ValueError(
            f"expected 6-class head, got weight={tuple(old_w.shape)} "
            f"bias={tuple(old_b.shape)}"
        )

    current = model.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in current and current[k].shape == v.shape and not k.startswith("classifier.1.")
    }
    model.load_state_dict(compatible, strict=False)

    head = model.classifier[1]
    if not isinstance(head, nn.Linear) or head.out_features != N_CLASSES:
        raise TypeError("model.classifier[1] must be a 7-class nn.Linear head")

    nn.init.kaiming_uniform_(head.weight, a=np.sqrt(5))
    if head.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(head.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(head.bias, -bound, bound)

    with torch.no_grad():
        head.weight[:5].copy_(old_w[:5])
        head.bias[:5].copy_(old_b[:5])
        head.weight[6].copy_(old_w[5])
        head.bias[6].copy_(old_b[5])


def build_optimizer(model: CNNGRUAttention, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": model.cnn.parameters(), "lr": args.backbone_lr_warmup, "name": "backbone"},
            {"params": model.gru.parameters(), "lr": args.attention_lr, "name": "gru"},
            {"params": model.attention.parameters(), "lr": args.attention_lr, "name": "attention"},
            {"params": model.classifier.parameters(), "lr": args.head_lr, "name": "head"},
        ],
        weight_decay=args.weight_decay,
    )


def maybe_finish_warmup(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_epochs: int,
    backbone_lr: float,
) -> None:
    if epoch != warmup_epochs + 1:
        return
    for group in optimizer.param_groups:
        if group.get("name") == "backbone":
            old_lr = group["lr"]
            group["lr"] = backbone_lr
            LOG.info("warmup complete: backbone lr %.2e -> %.2e", old_lr, backbone_lr)


def make_criterion(fall_weight: float, device: torch.device) -> nn.CrossEntropyLoss:
    weights = torch.ones(N_CLASSES, dtype=torch.float32, device=device)
    weights[0] = float(fall_weight)
    return nn.CrossEntropyLoss(weight=weights)


def train_one_epoch(
    model: CNNGRUAttention,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * yb.size(0)
        total_n += int(yb.size(0))
    return total_loss / max(total_n, 1)


def evaluate_logits(
    model: CNNGRUAttention,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    probs, labels, preds = [], [], []
    metas: list[dict] = []
    with torch.no_grad():
        for xb, yb, meta in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            prob = torch.softmax(logits, dim=1)
            total_loss += float(loss.item()) * yb.size(0)
            total_n += int(yb.size(0))
            probs.append(prob.cpu().numpy())
            labels.append(yb.cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            metas.append(meta)

    if not probs:
        return 0.0, np.empty((0, N_CLASSES)), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), metas
    return (
        total_loss / max(total_n, 1),
        np.concatenate(probs, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(preds, axis=0),
        metas,
    )


def predict_with_fall_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a global fall threshold while preserving argmax for non-fall cases."""
    if probs.size == 0:
        return np.empty((0,), dtype=np.int64)
    pred = probs.argmax(axis=1).astype(np.int64, copy=True)
    fall_prob = probs[:, 0]
    pred[fall_prob >= threshold] = 0
    nonfall_mask = (fall_prob < threshold) & (pred == 0)
    if nonfall_mask.any():
        pred[nonfall_mask] = probs[nonfall_mask, 1:].argmax(axis=1) + 1
    return pred


def evaluate_with_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> FallMetrics:
    """Thresholded metric calculation.

    TODO: run_eval / evaluate_with_threshold: replace this lightweight adapter
    with the finalized metrics function in a separate task if threshold policy
    needs extra reporting fields.
    """
    y_pred = predict_with_fall_threshold(probs, threshold)
    return compute_metrics(
        y_true,
        y_pred,
        classes=FINETUNE_CLASSES,
        recall_target=OPERATING_RECALL_TARGET,
        far_target=OPERATING_FAR_TARGET,
        f1_target=OPERATING_F1_TARGET,
    )


def select_global_threshold(y_true: np.ndarray, probs: np.ndarray) -> ThresholdResult:
    thresholds = np.round(np.arange(0.30, 0.7001, 0.05), 2)
    scored = [(float(t), evaluate_with_threshold(y_true, probs, float(t))) for t in thresholds]

    stretch = [
        (t, m) for t, m in scored
        if m.fall_recall >= STRETCH_RECALL_TARGET and m.far <= STRETCH_FAR_TARGET
    ]
    if stretch:
        t, m = max(stretch, key=lambda item: (item[1].fall_f1, -item[1].far, item[1].fall_recall))
        return ThresholdResult(threshold=t, metrics=m, rule="stretch")

    operating = [
        (t, m) for t, m in scored
        if m.fall_recall >= OPERATING_RECALL_TARGET and m.far <= OPERATING_FAR_TARGET
    ]
    if operating:
        t, m = max(operating, key=lambda item: (item[1].fall_f1, -item[1].far, item[1].fall_recall))
        return ThresholdResult(threshold=t, metrics=m, rule="operating")

    def fallback_key(item: tuple[float, FallMetrics]) -> tuple[float, float, float]:
        _, m = item
        far_over = max(0.0, m.far - OPERATING_FAR_TARGET)
        return (m.fall_recall, -far_over, m.fall_f1)

    t, m = max(scored, key=fallback_key)
    return ThresholdResult(threshold=t, metrics=m, rule="recall_first_far_over", exceeded_far=True)


def is_better_operating(candidate: EpochResult, best: EpochResult | None) -> bool:
    if best is None:
        return True
    c = candidate.primary_val
    b = best.primary_val
    c_ok = c.fall_recall >= OPERATING_RECALL_TARGET and c.far <= OPERATING_FAR_TARGET
    b_ok = b.fall_recall >= OPERATING_RECALL_TARGET and b.far <= OPERATING_FAR_TARGET
    if c_ok != b_ok:
        return c_ok
    return (
        c.fall_f1,
        -c.far,
        -candidate.primary_val_loss,
    ) > (
        b.fall_f1,
        -b.far,
        -best.primary_val_loss,
    )


def _serializable_args(args: argparse.Namespace) -> dict:
    """Convert Namespace to a dict safe for torch.load(weights_only=True).

    Path objects are converted to plain strings — pathlib.WindowsPath /
    PosixPath are not in the default safe-globals list, and saving them
    breaks reloads on PyTorch >= 2.6.
    """
    out: dict = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def save_checkpoint(
    path: Path,
    model: CNNGRUAttention,
    optimizer: torch.optim.Optimizer,
    epoch_result: EpochResult,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_result.epoch,
            "classes": list(FINETUNE_CLASSES),
            "threshold": epoch_result.threshold.threshold,
            "threshold_rule": epoch_result.threshold.rule,
            "primary_val_metrics": asdict(epoch_result.primary_val),
            "args": _serializable_args(args),
        },
        path,
    )


def false_positive_by_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    nonfall = y_true != 0
    fp_total = int(((y_pred == 0) & nonfall).sum())
    if fp_total == 0:
        return {name: 0.0 for name in FINETUNE_CLASSES[1:]}
    for idx, name in enumerate(FINETUNE_CLASSES[1:], start=1):
        cls_fp = int(((y_true == idx) & (y_pred == 0)).sum())
        out[name] = cls_fp / fp_total
    return out


def fold_flags(metrics: FallMetrics) -> list[str]:
    flags = []
    if metrics.far > 0.20:
        flags.append("HIGH_RISK_FAR")
    elif metrics.far > 0.10:
        flags.append("WARN_FAR")
    if metrics.fall_recall < 0.75:
        flags.append("HIGH_RISK_RECALL")
    elif metrics.fall_recall < 0.90:
        flags.append("WARN_RECALL")
    return flags


def summarize_fold_result(fold: int, threshold: float, y_true: np.ndarray, probs: np.ndarray) -> FoldReport:
    y_pred = predict_with_fall_threshold(probs, threshold)
    metrics = compute_metrics(
        y_true,
        y_pred,
        classes=FINETUNE_CLASSES,
        recall_target=OPERATING_RECALL_TARGET,
        far_target=OPERATING_FAR_TARGET,
        f1_target=OPERATING_F1_TARGET,
    )
    return FoldReport(
        fold=fold,
        threshold=threshold,
        metrics=metrics,
        flags=fold_flags(metrics),
        false_positive_by_class=false_positive_by_class(y_true, y_pred),
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    sampler: WeightedRandomSampler | None = None,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    LOG.info("device=%s seed=%d", device, args.seed)

    safesignal = SafeSignalDataset.from_npz(args.safesignal_cache)
    alsaify = AlsaifyDataset.from_npz(args.alsaify_cache)
    LOG.info("loaded SafeSignal=%d Alsaify=%d", len(safesignal), len(alsaify))

    if args.auto_sampler_preset:
        preset_source, preset_hard = infer_sampler_preset(len(safesignal))
        source_ratio = preset_source
        hard_weight = preset_hard
    else:
        source_ratio = args.source_ratio
        hard_weight = args.hard_weight
    LOG.info("sampler source_ratio SafeSignal=%.2f hard_weight=%.2f", source_ratio, hard_weight)

    ss_train, ss_val, ss_test = split_safesignal_by_fold(
        safesignal, fold=args.fold, val_ratio=args.val_ratio, seed=args.seed
    )
    alsaify_train, alsaify_val = split_alsaify_subject_holdout(
        alsaify, val_ratio=args.alsaify_val_ratio, seed=args.seed
    )

    # Augmentation hook: train subset only ([D-019] scope). Currently a no-op;
    # see TrainAugmentDataset TODO for the SafeSignal augmentation contract.
    ss_train_aug = TrainAugmentDataset(ss_train, augment=args.augment, base_seed=args.seed)
    train_dataset = ConcatDataset([alsaify_train, ss_train_aug])
    auxiliary_val = ConcatDataset([alsaify_val, ss_val])
    sample_weights = build_sample_weights(train_dataset, source_ratio, hard_weight)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = make_loader(train_dataset, args.batch_size, sampler=sampler, num_workers=args.num_workers)
    primary_val_loader = make_loader(ss_val, args.batch_size, num_workers=args.num_workers)
    auxiliary_val_loader = make_loader(auxiliary_val, args.batch_size, num_workers=args.num_workers)
    test_loader = make_loader(ss_test, args.batch_size, num_workers=args.num_workers)

    model = CNNGRUAttention(n_classes=N_CLASSES).to(device)
    if args.pretrained_ckpt is not None:
        migrate_pretrained_head(model, args.pretrained_ckpt, device)
        LOG.info("loaded pretrained 6-class checkpoint with 7-class head migration")
    optimizer = build_optimizer(model, args)
    criterion = make_criterion(args.fall_weight, device)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_operating: EpochResult | None = None
    best_val_loss: EpochResult | None = None
    stale_epochs = 0
    history: list[dict] = []
    result: EpochResult | None = None

    if args.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {args.epochs}")

    for epoch in range(1, args.epochs + 1):
        ss_train_aug.set_epoch(epoch)
        maybe_finish_warmup(optimizer, epoch, args.warmup_epochs, args.backbone_lr)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        primary_loss, primary_probs, primary_y, _, _ = evaluate_logits(
            model, primary_val_loader, criterion, device
        )
        aux_loss, aux_probs, aux_y, _, _ = evaluate_logits(
            model, auxiliary_val_loader, criterion, device
        )
        threshold = select_global_threshold(primary_y, primary_probs)
        primary_metrics = threshold.metrics
        aux_metrics = evaluate_with_threshold(aux_y, aux_probs, threshold.threshold)

        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            primary_val_loss=primary_loss,
            primary_val=primary_metrics,
            auxiliary_val_loss=aux_loss,
            auxiliary_val=aux_metrics,
            threshold=threshold,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "primary_val_loss": primary_loss,
                "threshold": threshold.threshold,
                "threshold_rule": threshold.rule,
                "primary_val": asdict(primary_metrics),
                "auxiliary_val": asdict(aux_metrics),
            }
        )

        LOG.info(
            "epoch=%03d train_loss=%.4f val_loss=%.4f threshold=%.2f/%s "
            "safe_val R=%.3f F1=%.3f FAR=%.3f aux R=%.3f F1=%.3f FAR=%.3f",
            epoch,
            train_loss,
            primary_loss,
            threshold.threshold,
            threshold.rule,
            primary_metrics.fall_recall,
            primary_metrics.fall_f1,
            primary_metrics.far,
            aux_metrics.fall_recall,
            aux_metrics.fall_f1,
            aux_metrics.far,
        )

        save_checkpoint(args.ckpt_dir / "last.pt", model, optimizer, result, args)

        if best_val_loss is None or primary_loss < best_val_loss.primary_val_loss:
            best_val_loss = result
            save_checkpoint(args.ckpt_dir / "best_val_loss.pt", model, optimizer, result, args)

        if epoch > args.warmup_epochs and is_better_operating(result, best_operating):
            best_operating = result
            stale_epochs = 0
            save_checkpoint(args.ckpt_dir / "best_operating.pt", model, optimizer, result, args)
            LOG.info("updated best_operating.pt")
        elif epoch >= args.early_stop_start:
            stale_epochs += 1

        if epoch >= args.early_stop_start and stale_epochs >= args.patience:
            LOG.info("early stopping: stale_epochs=%d patience=%d", stale_epochs, args.patience)
            break

    (args.ckpt_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_operating_path = args.ckpt_dir / "best_operating.pt"
    if best_operating is None:
        if result is None:
            raise RuntimeError("training loop produced no epoch result; cannot run sealed test")
        LOG.warning("best_operating was not selected; using last epoch threshold for test summary")
        operating_threshold = result.threshold.threshold
    else:
        operating_threshold = best_operating.threshold.threshold
        ckpt = torch.load(best_operating_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        LOG.info("loaded best_operating.pt for sealed test evaluation")

    _, test_probs, test_y, _, _ = evaluate_logits(model, test_loader, criterion, device)
    fold_report = summarize_fold_result(args.fold, operating_threshold, test_y, test_probs)
    report = asdict(fold_report)
    report["pass_fail"] = {
        "official": (
            fold_report.metrics.fall_recall >= OPERATING_RECALL_TARGET
            and fold_report.metrics.far <= OPERATING_FAR_TARGET
            and fold_report.metrics.fall_f1 >= OPERATING_F1_TARGET
        ),
        "stretch": (
            fold_report.metrics.fall_recall >= STRETCH_RECALL_TARGET
            and fold_report.metrics.far <= STRETCH_FAR_TARGET
        ),
    }
    (args.ckpt_dir / f"fold{args.fold}_test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOG.info(
        "test fold=%d threshold=%.2f R=%.3f F1=%.3f FAR=%.3f flags=%s",
        args.fold,
        operating_threshold,
        fold_report.metrics.fall_recall,
        fold_report.metrics.fall_f1,
        fold_report.metrics.far,
        ",".join(fold_report.flags) or "OK",
    )
    LOG.info("false positive pattern: %s", fold_report.false_positive_by_class)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeSignal combined fine-tuning")
    parser.add_argument("--safesignal_cache", type=Path, required=True)
    parser.add_argument("--alsaify_cache", type=Path, required=True)
    parser.add_argument("--pretrained_ckpt", type=Path, default=None)
    parser.add_argument("--ckpt_dir", type=Path, default=_PROJECT_ROOT / "model" / "finetune" / "checkpoints")
    parser.add_argument("--fold", type=int, choices=sorted(FOLDS), default=1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--alsaify_val_ratio", type=float, default=0.2)
    parser.add_argument("--source_ratio", type=float, default=0.60, help="SafeSignal target ratio.")
    parser.add_argument("--hard_weight", type=float, default=1.30, help="SafeSignal hard non-fall class weight.")
    parser.add_argument(
        "--auto_sampler_preset",
        action="store_true",
        help="Infer source_ratio/hard_weight from SafeSignal sample count presets.",
    )
    parser.add_argument("--fall_weight", type=float, default=1.0, choices=[1.0, 1.2, 1.5])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--backbone_lr_warmup", type=float, default=1e-5)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--attention_lr", type=float, default=3e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--early_stop_start", type=int, default=10)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Enable SafeSignal train-only on-the-fly augmentation: per sample/epoch, "
            "draw one of jittering/scaling/time_warping/noise_scale uniformly "
            "(deterministic in (seed, idx, epoch)). Alsaify samples pass through untouched."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
