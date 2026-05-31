"""SafeSignal CSV를 모델 입력 cache(.npz)로 변환한다.

수집 완료 후 실행하는 준비 단계다. 원본 CSV는 수정하지 않고, 보통
``data/cleaned``의 timestamp 정렬본을 입력으로 사용한다.

Class policy:
  - pretrained6: baseline best.pt 평가용. RUN/NO_MOTION 제외, picking=5.
  - finetune7: 현재 model/finetune/train.py 호환. RUN 포함, NO_MOTION 제외.
  - no_motion8: NO_MOTION을 class로 포함하는 실험용. 현재 train.py와는 미호환.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from collect.labels import ACTIVITY_INFO
from model.preprocessing.loader import parse_safesignal_filename
from model.preprocessing.pipeline import preprocess_safesignal_file_full


CLASS_POLICIES: dict[str, tuple[str, ...]] = {
    "pretrained6": ("fall", "walking", "sit_stand", "lying", "standing", "picking"),
    "finetune7": ("fall", "walking", "sit_stand", "lying", "standing", "running", "picking"),
    "no_motion8": (
        "fall",
        "walking",
        "sit_stand",
        "lying",
        "standing",
        "running",
        "picking",
        "no_motion",
    ),
}

ACTIVITY_TO_NAME: dict[str, str] = {
    code: info["display"] for code, info in ACTIVITY_INFO.items()
}

ACTIVITY_TO_CLASS: dict[str, str] = {
    "FALL_SIT_F": "fall",
    "FALL_SIT_B": "fall",
    "FALL_STD_F": "fall",
    "FALL_STD_B": "fall",
    "FALL_WALK_F": "fall",
    "FALL_WALK_B": "fall",
    "WALK": "walking",
    "SIT_STD": "sit_stand",
    "LIE": "lying",
    "STAND": "standing",
    "RUN": "running",
    "PICK": "picking",
    "NO_MOTION": "no_motion",
}


@dataclass(frozen=True)
class CacheSummary:
    policy: str
    classes: list[str]
    source_dir: str
    out: str
    files_seen: int
    files_used: int
    windows: int
    skipped: dict[str, int]
    class_counts: dict[str, int]
    subjects: list[int]
    envs: list[int]


def _label_for_activity(activity: str, classes: tuple[str, ...]) -> int | None:
    class_name = ACTIVITY_TO_CLASS.get(activity)
    if class_name is None or class_name not in classes:
        return None
    return classes.index(class_name)


def _find_csvs(data_dir: Path, envs: set[int] | None, subjects: set[int] | None) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(data_dir.glob("*.csv")):
        try:
            meta = parse_safesignal_filename(path)
        except ValueError:
            continue
        if envs is not None and meta.environment not in envs:
            continue
        if subjects is not None and meta.subject not in subjects:
            continue
        paths.append(path)
    return paths


def build_cache(args: argparse.Namespace) -> CacheSummary:
    data_dir = (PROJECT_ROOT / args.dir).resolve()
    out_path = (PROJECT_ROOT / args.out).resolve()
    classes = CLASS_POLICIES[args.policy]
    envs = set(args.env) if args.env else None
    subjects = set(args.subject) if args.subject else None
    activities = {str(a).upper() for a in args.activity} if args.activity else None
    csvs = [
        path for path in _find_csvs(data_dir, envs, subjects)
        if (activities is None or parse_safesignal_filename(path).activity in activities)
        and _label_for_activity(parse_safesignal_filename(path).activity, classes) is not None
    ]
    if args.max_files is not None:
        csvs = csvs[: args.max_files]

    Xs: list[np.ndarray] = []
    ys: list[int] = []
    subject_values: list[int] = []
    env_values: list[int] = []
    filenames: list[str] = []
    activities: list[str] = []
    trials: list[int] = []
    skipped: dict[str, int] = {}
    files_used = 0

    for idx, csv in enumerate(csvs, start=1):
        meta = parse_safesignal_filename(csv)
        label = _label_for_activity(meta.activity, classes)
        if label is None:
            raise AssertionError("csvs should be filtered by class policy before processing")
        try:
            result = preprocess_safesignal_file_full(
                csv,
                rx=args.rx,
                max_gap_ms=args.max_gap_ms,
                stride=args.stride,
                tail_window=args.tail_window,
                pad_short=args.pad_short,
                rpca_max_iter=args.rpca_max_iter,
                rpca_tol=args.rpca_tol,
            )
        except Exception as exc:
            skipped[type(exc).__name__] = skipped.get(type(exc).__name__, 0) + 1
            print(f"[skip] {csv.name}: {exc}")
            continue

        n = int(result.inputs.shape[0])
        if n == 0:
            skipped["empty_windows"] = skipped.get("empty_windows", 0) + 1
            print(f"[skip] {csv.name}: empty_windows")
            continue

        files_used += 1
        Xs.append(result.inputs.astype(np.float32, copy=False))
        ys.extend([label] * n)
        subject_values.extend([meta.subject] * n)
        env_values.extend([meta.environment] * n)
        filenames.extend([meta.filename] * n)
        activities.extend([meta.activity] * n)
        trials.extend([meta.trial] * n)

        if args.progress_every and idx % args.progress_every == 0:
            print(f"[progress] {idx}/{len(csvs)} files, windows={len(ys)}")

    if not Xs:
        raise SystemExit("FAIL: cache에 저장할 window가 없습니다.")

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    subject_arr = np.asarray(subject_values, dtype=np.int64)
    env_arr = np.asarray(env_values, dtype=np.int64)
    filename_arr = np.asarray(filenames, dtype=object)
    activity_arr = np.asarray(activities, dtype=object)
    trial_arr = np.asarray(trials, dtype=np.int64)
    source_arr = np.asarray(["safesignal"] * len(y), dtype=object)
    is_augmented = np.zeros(len(y), dtype=bool)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"FAIL: output exists: {out_path} (--overwrite 필요)")

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        subject=subject_arr,
        source=source_arr,
        env=env_arr,
        filename=filename_arr,
        activity=activity_arr,
        trial=trial_arr,
        classes=np.asarray(classes, dtype=object),
        class_policy=args.policy,
        is_augmented=is_augmented,
    )

    class_counts = {
        name: int((y == i).sum())
        for i, name in enumerate(classes)
    }
    summary = CacheSummary(
        policy=args.policy,
        classes=list(classes),
        source_dir=str(data_dir),
        out=str(out_path),
        files_seen=len(csvs),
        files_used=files_used,
        windows=int(len(y)),
        skipped=skipped,
        class_counts=class_counts,
        subjects=sorted({int(v) for v in subject_arr.tolist()}),
        envs=sorted({int(v) for v in env_arr.tolist()}),
    )

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    print(f"[write] {out_path}")
    print(f"[write] {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeSignal CSV -> model cache(.npz)")
    parser.add_argument("--dir", default="data/cleaned", help="CSV 입력 폴더")
    parser.add_argument("--out", required=True, help="출력 .npz 경로")
    parser.add_argument("--policy", choices=sorted(CLASS_POLICIES), default="finetune7")
    parser.add_argument("--env", nargs="+", type=int, default=None, help="환경 필터 예: 1 4")
    parser.add_argument("--subject", nargs="+", type=int, default=None, help="subject 필터 예: 1 2 3")
    parser.add_argument("--activity", nargs="+", default=None, help="활동 필터 예: WALK FALL_SIT_F")
    parser.add_argument("--rx", choices=["rx1", "rx2", "both"], default="both")
    parser.add_argument("--stride", type=int, default=None, help="window stride. 기본=300")
    parser.add_argument(
        "--tail-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="잔여 구간 tail window 추가 (기본 true)",
    )
    parser.add_argument("--pad-short", action="store_true", help="짧은 세션 zero padding")
    parser.add_argument("--max-gap-ms", type=float, default=100.0)
    parser.add_argument("--rpca-max-iter", type=int, default=200)
    parser.add_argument("--rpca-tol", type=float, default=None)
    parser.add_argument("--max-files", type=int, default=None, help="dry-run용 파일 수 제한")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_cache(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
