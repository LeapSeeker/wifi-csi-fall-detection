"""Alsaify LOS (E1+E2) 사전학습 루프.

워크플로
  1) CSV 스캔 → 활동 코드를 6 클래스로 매핑 → 캐시 빌드(.npz, 1회 ~수십 분)
  2) 캐시 로드 → subject 단위 8:2 split → DataLoader
  3) CNNGRUAttention 학습 → epoch마다 fall recall/precision/F1 추적 → best/last 저장

실행 예
  python -m model.pretrained.train --epochs 30 --batch_size 16 --lr 1e-3
  python model/pretrained/train.py --epochs 30
  python -m model.pretrained.train --rebuild_cache       # 캐시 재빌드
  python -m model.pretrained.train --device cpu          # GPU 없을 때

MX450 (2GB VRAM): --batch_size 8~16 권장. 모델 자체가 387K 파라미터로 작아 OOM 가능성 낮음.

선결 조건
  data/alsaify-raw/Environment {1,2}/Subject_*/ 아래 CSV가 존재해야 함
  (배포 zip 상태면 먼저 압축 해제 필요).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# 프로젝트 루트를 sys.path에 보장 (직접 실행 + ProcessPoolExecutor 워커 모두 대응).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model.preprocessing import parse_alsaify_filename, preprocess_files_full
from model.pretrained.metrics import compute_metrics, format_report, save_metrics
from model.pretrained.model import CLASSES, CNNGRUAttention, count_parameters

# ── 설정 ────────────────────────────────────────────────────────────────────
# Alsaify 활동 코드 → 6 클래스 인덱스 (model/CLAUDE.md 사전학습 매핑 표 준수)
ACTIVITY_TO_LABEL: dict[int, int] = {
    2: 0, 5: 0,    # fall
    6: 1, 8: 1,    # walking
    10: 2, 11: 2,  # sit_stand
    3: 3,          # lying (C1+C2 통합)
    4: 4,          # standing
    12: 5,         # picking
}
LOS_ENVS: tuple[int, ...] = (1, 2)  # E3(NLOS) 제외

DEFAULT_DATA_ROOT = _PROJECT_ROOT / "data" / "alsaify-raw"
DEFAULT_CKPT_DIR = _PROJECT_ROOT / "model" / "pretrained" / "checkpoints"
DEFAULT_CACHE_PATH = DEFAULT_CKPT_DIR / "dataset_cache.npz"


# ── 캐시 빌드 ──────────────────────────────────────────────────────────────
def _find_csvs(data_root: Path, envs: tuple[int, ...] = LOS_ENVS) -> list[Path]:
    """지정 환경의 매핑 가능 활동 코드 CSV만 수집. 기본 LOS_ENVS=(1,2)."""
    found = []
    for env in envs:
        env_dir = data_root / f"Environment {env}"
        if not env_dir.exists():
            continue
        for csv in env_dir.glob("**/*.csv"):
            try:
                meta = parse_alsaify_filename(csv)
            except ValueError:
                continue
            if meta.environment != env:
                continue
            if meta.activity not in ACTIVITY_TO_LABEL:
                continue
            found.append(csv)
    return sorted(found)


def _print_class_counts(y: np.ndarray) -> None:
    print("클래스 분포:")
    for i, name in enumerate(CLASSES):
        n = int((y == i).sum())
        print(f"  [{i}] {name:10s}: {n}")


def build_cache(
    data_root: Path,
    cache_path: Path,
    n_jobs: int,
    envs: tuple[int, ...] = LOS_ENVS,
) -> None:
    csvs = _find_csvs(data_root, envs=envs)
    if not csvs:
        raise SystemExit(
            f"FAIL: {data_root} 아래 envs={envs} CSV가 없습니다. "
            "Subject_*.zip이 압축 해제되어 있는지 확인하세요."
        )
    env_str = "+".join(f"E{e}" for e in envs)
    print(f"발견된 CSV: {len(csvs)}개  ({env_str}, 매핑 가능 활동만)")

    # n_jobs 관례: -1 → cpu_count()-1 (None으로 변환)
    n_workers = None if n_jobs <= 0 else n_jobs
    print(f"전처리 시작 (CSV 1개당 ~2초, n_workers={n_workers or 'auto'})... → {cache_path}")

    t0 = time.time()
    results = preprocess_files_full(csvs, n_workers=n_workers)
    print(f"전처리 완료 [{time.time() - t0:.1f}s]")

    Xs, ys, subs = [], [], []
    for r in results:
        if r.inputs.shape[0] == 0:
            continue
        label = ACTIVITY_TO_LABEL[r.meta.activity]
        Xs.append(r.inputs)
        ys.extend([label] * r.inputs.shape[0])
        subs.extend([r.meta.subject] * r.inputs.shape[0])

    if not Xs:
        raise SystemExit("FAIL: 전처리 결과가 비었습니다.")

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    subject = np.asarray(subs, dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, subject=subject)
    print(f"캐시 저장: {cache_path}")
    print(f"  X={X.shape}  y={y.shape}  subjects={sorted(set(subject.tolist()))}")
    _print_class_counts(y)


# ── 데이터 로딩 / split ────────────────────────────────────────────────────
@dataclass
class Splits:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    train_subjects: list[int]
    val_subjects: list[int]


def load_and_split(cache_path: Path, val_ratio: float, seed: int) -> Splits:
    npz = np.load(cache_path)
    X, y, subject = npz["X"], npz["y"], npz["subject"]
    print(f"\n캐시 로드: X={X.shape}  y={y.shape}  "
          f"subjects={len(set(subject.tolist()))}개")
    _print_class_counts(y)

    rng = np.random.default_rng(seed)
    subjects = sorted(set(int(s) for s in subject.tolist()))
    if len(subjects) < 2:
        raise SystemExit(
            f"FAIL: subject 단위 split을 위해 ≥2개 subject 필요 (현재 {subjects}). "
            "추가 Subject_*.zip을 압축 해제하세요."
        )
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_ratio)))
    val_subs = sorted(subjects[:n_val])
    train_subs = sorted(subjects[n_val:])

    train_mask = np.isin(subject, train_subs)
    val_mask = np.isin(subject, val_subs)
    print(f"split — train subjects {train_subs} ({int(train_mask.sum())} samples)")
    print(f"      — val   subjects {val_subs} ({int(val_mask.sum())} samples)")

    return Splits(
        X_train=torch.from_numpy(X[train_mask]),
        y_train=torch.from_numpy(y[train_mask]),
        X_val=torch.from_numpy(X[val_mask]),
        y_val=torch.from_numpy(y[val_mask]),
        train_subjects=train_subs,
        val_subjects=val_subs,
    )


# ── 학습 / 평가 ────────────────────────────────────────────────────────────
@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    fall_recall: float
    fall_precision: float
    fall_f1: float


def evaluate(model, loader, device, criterion):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            losses.append(criterion(logits, yb).item() * yb.size(0))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(yb.cpu().numpy())
    n = sum(p.shape[0] for p in preds)
    avg_loss = sum(losses) / max(n, 1)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    acc = float((y_pred == y_true).mean())
    return avg_loss, acc, y_true, y_pred


def train(
    splits: Splits,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    ckpt_dir: Path,
    num_workers: int,
) -> None:
    train_loader = DataLoader(
        TensorDataset(splits.X_train, splits.y_train),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(splits.X_val, splits.y_val),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    model = CNNGRUAttention().to(device)
    print(f"\nmodel params: {count_parameters(model):,}  device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: list[EpochMetrics] = []
    best_recall = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = ep_n = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * yb.size(0)
            ep_n += yb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = ep_loss / max(ep_n, 1)

        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device, criterion)
        m = compute_metrics(y_true, y_pred, classes=list(CLASSES))

        em = EpochMetrics(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            val_acc=m.accuracy, fall_recall=m.fall_recall,
            fall_precision=m.fall_precision, fall_f1=m.fall_f1,
        )
        history.append(em)
        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={m.accuracy:.3f}  "
            f"fall: R={m.fall_recall:.3f} P={m.fall_precision:.3f} "
            f"F1={m.fall_f1:.3f} FAR={m.far:.3f}  "
            f"[{time.time() - t0:.1f}s]"
        )

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": asdict(em),
            "classes": list(CLASSES),
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if m.fall_recall > best_recall:
            best_recall = m.fall_recall
            torch.save(ckpt, ckpt_dir / "best.pt")
            save_metrics(m, ckpt_dir / "best_metrics.json")
            print(f"  ↑ best fall recall {m.fall_recall:.3f} → {ckpt_dir/'best.pt'}")

    (ckpt_dir / "history.json").write_text(
        json.dumps([asdict(em) for em in history], indent=2)
    )
    # 마지막 epoch의 풀 리포트 (혼동 행렬 + 목표 달성)
    save_metrics(m, ckpt_dir / "final_metrics.json")
    print(f"\nhistory saved: {ckpt_dir/'history.json'}")
    print(format_report(m, header="\n[Final Validation Report]"))


# ── main ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32,
                   help="MX450(2GB) → 8~16 권장. 모델 작아 OOM 가능성 낮음.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--cache_path", type=Path, default=None,
                   help="기본: checkpoints/dataset_cache[_e<envs>].npz "
                        "(envs!=(1,2)이면 자동 접미사)")
    p.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--envs", type=int, nargs="+", default=list(LOS_ENVS),
                   choices=[1, 2, 3],
                   help="사용할 환경. 기본 1 2 (LOS). 빠른 검증: --envs 1")
    p.add_argument("--rebuild_cache", action="store_true",
                   help="기존 캐시를 무시하고 재빌드")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader 워커. Windows는 0 권장.")
    p.add_argument("--n_jobs", type=int, default=-1,
                   help="캐시 빌드 병렬 워커 수 (-1 = all cores)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    envs = tuple(sorted(set(args.envs)))
    if args.cache_path is None:
        suffix = "" if envs == LOS_ENVS else "_e" + "".join(str(e) for e in envs)
        args.cache_path = DEFAULT_CKPT_DIR / f"dataset_cache{suffix}.npz"

    if args.rebuild_cache or not args.cache_path.exists():
        build_cache(args.data_root, args.cache_path, n_jobs=args.n_jobs, envs=envs)
    else:
        print(f"기존 캐시 사용: {args.cache_path}  (--rebuild_cache로 재빌드)")

    splits = load_and_split(args.cache_path, val_ratio=args.val_ratio, seed=args.seed)
    if splits.X_train.shape[0] == 0 or splits.X_val.shape[0] == 0:
        print("FAIL: train/val 샘플이 비었습니다. data_root 확인 필요.")
        return 1

    device = resolve_device(args.device)
    train(
        splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        ckpt_dir=args.ckpt_dir,
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
