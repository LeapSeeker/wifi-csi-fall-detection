"""기존 SafeSignal CSV를 timestamp_us 기준으로 정렬해 별도 폴더에 저장한다.

원본 data/raw 파일은 수정하지 않는다. 페어가 확정된 도착 순서가 아니라
물리적 측정 시각(timestamp_us) 순서로 time-series row를 재정렬한 사본을 만든다.

실행:
    python debug/data_collect/sort_csv_by_timestamp.py --dir data/raw --out data/cleaned
    python debug/data_collect/sort_csv_by_timestamp.py --dir data/raw --out data/cleaned --env E2 E4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def sort_one(src: Path, dst: Path) -> tuple[bool, str]:
    try:
        df = pd.read_csv(src)
    except Exception as exc:
        return False, f"read failed: {exc}"

    if "timestamp_us" not in df.columns:
        return False, "missing timestamp_us"

    if len(df) > 1:
        arrival = pd.Series(range(len(df)), index=df.index, name="_arrival_idx")
        df = pd.concat([df, arrival], axis=1)
        df = df.sort_values(
            ["timestamp_us", "_arrival_idx"],
            kind="stable",
        ).drop(columns=["_arrival_idx"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return True, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeSignal CSV timestamp 정렬본 생성")
    parser.add_argument("--dir", default="data/raw", help="원본 CSV 폴더")
    parser.add_argument("--out", default="data/cleaned", help="정렬본 출력 폴더")
    parser.add_argument("--env", nargs="+", default=None, help="환경 필터 예: E2 E4")
    parser.add_argument("--overwrite", action="store_true", help="기존 출력 파일 덮어쓰기")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = (PROJECT_ROOT / args.dir).resolve()
    out_dir = (PROJECT_ROOT / args.out).resolve()

    if not src_dir.exists():
        raise SystemExit(f"[ERROR] source dir not found: {src_dir}")

    csvs = sorted(src_dir.glob("*.csv"))
    if args.env:
        envs = {e.upper() for e in args.env}
        csvs = [p for p in csvs if any(p.name.upper().startswith(f"{env}_") for env in envs)]

    if not csvs:
        print(f"[INFO] no csv files: {src_dir}")
        return

    ok_count = 0
    skip_count = 0
    for src in csvs:
        dst = out_dir / src.name
        if dst.exists() and not args.overwrite:
            skip_count += 1
            print(f"[skip] exists: {dst.name}")
            continue
        ok, msg = sort_one(src, dst)
        if ok:
            ok_count += 1
            print(f"[ok] {src.name} -> {dst}")
        else:
            skip_count += 1
            print(f"[skip] {src.name}: {msg}")

    print(f"\nDONE sorted={ok_count} skipped={skip_count} out={out_dir}")


if __name__ == "__main__":
    main()
