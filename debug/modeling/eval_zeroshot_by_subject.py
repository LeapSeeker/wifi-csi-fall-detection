"""best.pt zero-shot 진단 — subject(fold)별 6-class 평가 + fall threshold sweep.

목적: SafeSignal fall recall 저하가 도메인 한계(pretrained가 SafeSignal에
약함)인지 fine-tuning 문제인지 가르기 위해, Alsaify로 사전학습된 best.pt를
SafeSignal pretrained6 cache에 zero-shot으로 적용한다. 학습 없음.

evaluate_baseline6.py를 참고했으나 별도 파일이며 기존 파일은 수정하지 않는다.
cache·텐서 모두 read-only (in-place 변형 없음). 결과는 stdout 출력만 한다.

판정: fall = CLASSES[0] (index 0). softmax fall 확률 > threshold 이면 fall로
판정하는 이진 결정으로 threshold 0.30~0.70 (step 0.05) sweep.
  fall_recall = TP / (TP + FN)
  far         = FP / (FP + TN)
  fall_f1     = 2PR / (P + R),  P = TP / (TP + FP)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 콘솔(cp949)에서 em-dash 등 비-cp949 문자 출력 시 깨지지 않도록.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.pretrained.model import CLASSES, CNNGRUAttention

FALL_LABEL = 0  # CLASSES[0] == "fall"
SWEEP = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30 .. 0.70
# subject 정수 → 보고용 라벨 (fold 매핑)
SUBJECT_LABELS = {1: "S01(fold1)", 2: "S02(fold2)", 3: "S03(fold3)"}


def _load_state(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def _binary_fall_metrics(fall_prob: np.ndarray, true_fall: np.ndarray, thr: float) -> dict:
    """fall_prob > thr 이진 판정으로 R/FAR/P/F1 직접 계산 (read-only)."""
    pred_fall = fall_prob > thr
    tp = int(np.sum(pred_fall & true_fall))
    fp = int(np.sum(pred_fall & ~true_fall))
    fn = int(np.sum(~pred_fall & true_fall))
    tn = int(np.sum(~pred_fall & ~true_fall))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    return {"thr": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "precision": precision, "f1": f1, "far": far}


def _best_recall_row(rows: list[dict]) -> dict:
    """fall_recall 최대 지점. 동률이면 f1 높은 쪽, 그래도 동률이면 far 낮은 쪽."""
    return max(rows, key=lambda r: (r["recall"], r["f1"], -r["far"]))


def evaluate(args: argparse.Namespace) -> None:
    cache = np.load(args.cache, allow_pickle=True)
    classes = [str(c) for c in cache["classes"].tolist()] if "classes" in cache else list(CLASSES)
    assert tuple(classes) == tuple(CLASSES), (
        f"cache classes != 6-class CLASSES\n  cache={classes}\n  expected={list(CLASSES)}"
    )
    assert "subject" in cache.files, "cache에 subject 키가 없습니다 (fold 분리 불가)"

    X = cache["X"].astype(np.float32, copy=False)
    y = cache["y"].astype(np.int64, copy=False)
    subject = cache["subject"].astype(np.int64, copy=False)

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = CNNGRUAttention(n_classes=len(CLASSES)).to(device)
    model.load_state_dict(_load_state(args.ckpt, device), strict=True)
    model.eval()

    # 전체 forward → fall 확률 (read-only, no_grad)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    fall_probs: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            fall_probs.append(p[:, FALL_LABEL])
    fall_prob = np.concatenate(fall_probs, axis=0)
    assert fall_prob.shape[0] == y.shape[0]

    # subject(fold)별 평가
    print("=" * 72)
    print(f"best.pt zero-shot — subject별 6-class  (ckpt={Path(args.ckpt).name}, device={device})")
    print(f"cache: {args.cache}  windows={len(y)}")
    print("=" * 72)

    per_subject: dict[int, dict] = {}
    for s in sorted(SUBJECT_LABELS):
        mask = subject == s
        n = int(mask.sum())
        if n == 0:
            continue
        true_fall = (y[mask] == FALL_LABEL)
        fp_s = fall_prob[mask]
        rows = [_binary_fall_metrics(fp_s, true_fall, thr) for thr in SWEEP]
        at030 = next(r for r in rows if r["thr"] == 0.30)
        best = _best_recall_row(rows)
        per_subject[s] = {
            "label": SUBJECT_LABELS[s], "n": n,
            "fall_windows": int(true_fall.sum()),
            "at030": at030, "best": best, "rows": rows,
        }

    # ── 보고 형식 출력 ──────────────────────────────────────────────
    labels = [per_subject[s]["label"] for s in sorted(per_subject)]
    col = "{:>14}"
    hdr = " " * 16 + "".join(col.format(l) for l in labels)
    print(hdr)

    def line(name: str, fn) -> str:
        return f"{name:<16}" + "".join(col.format(fn(per_subject[s])) for s in sorted(per_subject))

    print(line("fall windows", lambda d: d["fall_windows"]))
    print(line("@thr0.30 R", lambda d: f"{d['at030']['recall']:.3f}"))
    print(line("@thr0.30 FAR", lambda d: f"{d['at030']['far']:.3f}"))
    print(line("@thr0.30 F1", lambda d: f"{d['at030']['f1']:.3f}"))
    print()
    print("best-recall thr / R / FAR / F1 (subject별):")
    for s in sorted(per_subject):
        d = per_subject[s]
        b = d["best"]
        print(f"  {d['label']:<12} thr={b['thr']:.2f}  R={b['recall']:.3f}  "
              f"FAR={b['far']:.3f}  F1={b['f1']:.3f}")

    # 전체 sweep 표 (참고)
    print()
    print("[참고] subject별 threshold sweep (R / FAR / F1):")
    for s in sorted(per_subject):
        d = per_subject[s]
        print(f"  {d['label']} (n={d['n']}, fall={d['fall_windows']}):")
        for r in d["rows"]:
            print(f"    thr={r['thr']:.2f}  R={r['recall']:.3f}  "
                  f"FAR={r['far']:.3f}  F1={r['f1']:.3f}  "
                  f"(TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="best.pt zero-shot subject별 진단")
    p.add_argument("--cache", type=Path,
                   default=PROJECT_ROOT / "model" / "finetune" / "cache" / "safesignal_e1234_pretrained6.npz")
    p.add_argument("--ckpt", type=Path,
                   default=PROJECT_ROOT / "model" / "pretrained" / "checkpoints" / "best.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def main() -> int:
    evaluate(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
