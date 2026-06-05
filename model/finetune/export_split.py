"""현재 split 을 JSON 으로 내보내는 스크립트.

generate_dummy_fall_cache.py 에 train 세션 목록을 제공하기 위해
학습 전에 반드시 먼저 실행해야 한다.

사용:
    python -m model.finetune.export_split \\
        --cache model/finetune/cache/safesignal_e1234_pretrained6.npz \\
        --out   model/finetune/cache/split_seed42.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.finetune.train import SafeSignalDataset, split_safesignal_within_subject


def main() -> int:
    ap = argparse.ArgumentParser(description="split → JSON 내보내기")
    ap.add_argument("--cache", type=Path, required=True,
                    help="safesignal NPZ 캐시 경로")
    ap.add_argument("--out",   type=Path, required=True,
                    help="출력 JSON 경로")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--val_ratio",  type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    args = ap.parse_args()

    ds = SafeSignalDataset.from_npz(args.cache)
    tr, va, te = split_safesignal_within_subject(
        ds,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # 윈도우 인덱스 → 세션 파일명 (중복 제거)
    def session_names(subset) -> list[str]:
        return sorted(set(ds.filenames[i] for i in subset.indices))

    split = {
        "seed":       args.seed,
        "val_ratio":  args.val_ratio,
        "test_ratio": args.test_ratio,
        "cache":      str(args.cache),
        "train": session_names(tr),
        "val":   session_names(va),
        "test":  session_names(te),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"저장: {args.out}")
    print(f"train={len(split['train'])}  val={len(split['val'])}  "
          f"test={len(split['test'])}  (seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
