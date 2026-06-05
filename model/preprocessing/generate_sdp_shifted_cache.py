"""SDP feature 공간에서 fall 패턴을 시간 이동시켜 더미 학습 데이터 생성.

목적
----
현재 training 캐시(ss_peak160_p6.npz)의 fall 윈도우는 낙상 onset이
SDP 시간축 step 0~9 사이에만 위치한다.
추론 시 낙상이 윈도우 중반~후반(step 10~25)에 걸칠 경우 모델이
이 패턴을 학습한 적이 없으므로 recall이 떨어진다.

이 스크립트는 training fall 샘플을 SDP 시간축으로 우측 이동(right-shift)해
"fall이 늦게 시작되는 윈도우" 패턴을 합성한다.

규칙
----
- split_seed42.json 로 train 세션만 필터 (test 오염 방지)
- is_augmented=True 마킹
- filename = SHIFT{k}_{원본파일명}  (고유 ID 보장)
- 우측 이동 시 좌측 빈 구간은 pre-fall 배경으로 채움
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# SDP 시간 축 길이
T = 28

FALL_CLS = 0

# 우측 이동 스텝: 이 값만큼 fall이 윈도우 내에서 뒤로 밀린다
SHIFT_STEPS = [5, 10, 15, 20]

# 가우시안 노이즈 sigma (SDP 정규화 후 스케일 기준)
NOISE_SIGMA = 0.25


def _shift_right(x: np.ndarray, shift: int, bg: np.ndarray) -> np.ndarray:
    """(1, 28, 20) SDP feature를 시간축으로 오른쪽으로 shift 이동.

    Args:
        x: (1, T, F) SDP feature
        shift: 이동 스텝 수 (양수 = 우측)
        bg: (1, 1, F) 배경값 (빈 구간 채우기용)
    Returns:
        (1, T, F) 이동된 feature
    """
    if shift <= 0:
        return x.copy()
    out = np.empty_like(x)
    out[:, shift:, :] = x[:, :T - shift, :]
    out[:, :shift, :] = bg
    return out.astype(np.float32)


def build_shifted_cache(
    cache_path: Path,
    split_path: Path,
    out_path: Path,
    shift_steps: list[int],
    noise_sigma: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    # ── split 로드 ──────────────────────────────────────────────────────
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_filenames: set[str] = set(split["train"])
    print(f"split 로드: train={len(split['train'])}  val={len(split['val'])}  test={len(split['test'])}")

    # ── 캐시 로드 ────────────────────────────────────────────────────────
    d = np.load(cache_path, allow_pickle=True)
    X         = d["X"]           # (N, 1, 28, 20)
    y         = d["y"]           # (N,)
    filenames = d["filename"]    # (N,)  object
    subjects  = d["subject"]     # (N,)
    envs      = d["env"]         # (N,)
    activities= d["activity"]    # (N,)
    trials    = d["trial"]       # (N,)
    wfi       = d["within_file_index"]  # (N,)

    # ── train 낙상 샘플만 선택 ────────────────────────────────────────
    mask = (y == FALL_CLS) & np.array([fn in train_filenames for fn in filenames])
    idx  = np.where(mask)[0]
    print(f"train 낙상 샘플: {len(idx)}  (전체 낙상: {int((y==FALL_CLS).sum())})")

    # ── 배경값 추정: 비낙상 training 샘플 평균 ─────────────────────────
    nf_mask = (y != FALL_CLS) & np.array([fn in train_filenames for fn in filenames])
    bg_mean = X[nf_mask].mean(axis=0, keepdims=True)  # (1, 1, 28, 20) → broadcast later

    records: list[dict] = []

    for i in idx:
        x_orig = X[i]          # (1, 28, 20)
        fn     = filenames[i]
        subj   = int(subjects[i])
        env    = int(envs[i])
        act    = str(activities[i])
        trial_ = int(trials[i])
        w_idx  = int(wfi[i])

        # 배경: 원본의 첫 3 SDP 타임스텝 평균
        bg = x_orig[:, :3, :].mean(axis=1, keepdims=True)  # (1, 1, 20)

        for shift in shift_steps:
            if T - shift < 5:  # 너무 많이 이동하면 fall 신호가 거의 없음
                continue
            x_shifted = _shift_right(x_orig, shift, bg)
            # 약한 노이즈 추가
            x_shifted = x_shifted + rng.normal(0, noise_sigma, size=x_shifted.shape).astype(np.float32)
            records.append({
                "x":                 x_shifted,
                "y":                 FALL_CLS,
                "filename":          f"SHIFT{shift:02d}_{fn}",
                "origin_filename":   fn,
                "shift_step":        shift,
                "subject":           subj,
                "env":               env,
                "activity":          act,
                "trial":             trial_,
                "within_file_index": w_idx,
            })

    if not records:
        raise RuntimeError("생성된 샘플 없음")

    print(f"\n생성 완료: {len(records)} 샘플")
    print(f"  shift 분포: {dict(zip(shift_steps, [sum(1 for r in records if r['shift_step']==s) for s in shift_steps]))}")

    X_out   = np.stack([r["x"]                for r in records]).astype(np.float32)
    y_out   = np.array([r["y"]                for r in records], dtype=np.int64)
    fn_out  = np.array([r["filename"]         for r in records], dtype=object)
    orig_fn = np.array([r["origin_filename"]  for r in records], dtype=object)
    shift_a = np.array([r["shift_step"]       for r in records], dtype=np.int64)
    subj_o  = np.array([r["subject"]          for r in records], dtype=np.int64)
    env_o   = np.array([r["env"]              for r in records], dtype=np.int64)
    act_o   = np.array([r["activity"]         for r in records], dtype=object)
    trial_o = np.array([r["trial"]            for r in records], dtype=np.int64)
    wfi_o   = np.array([r["within_file_index"]for r in records], dtype=np.int64)
    is_aug  = np.ones(len(records), dtype=bool)
    source  = np.array(["safesignal_sdp_shift"] * len(records), dtype=object)

    CLASS_NAMES = d["classes"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X=X_out, y=y_out,
        filename=fn_out, origin_filename=orig_fn,
        shift_step=shift_a,
        subject=subj_o, source=source, env=env_o,
        activity=act_o, trial=trial_o, within_file_index=wfi_o,
        is_augmented=is_aug,
        classes=CLASS_NAMES,
        class_policy="pretrained6",
        normalization="global_self",
        build_params=np.array([
            f"shift_steps={shift_steps}",
            f"noise_sigma={noise_sigma}",
            f"seed={seed}",
        ], dtype=object),
    )
    print(f"저장: {out_path}")

    print("\n[ wfi별 샘플 수 ]")
    for w in sorted(set(wfi_o.tolist())):
        print(f"  wfi={w}: {int((wfi_o==w).sum())}")

    print("\n[ shift_step별 샘플 수 ]")
    for s in sorted(set(shift_a.tolist())):
        print(f"  shift={s}: {int((shift_a==s).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SDP temporal-shift 더미 낙상 캐시 생성")
    ap.add_argument("--cache", type=Path,
                    default=Path("model/finetune/cache/ss_peak160_p6.npz"))
    ap.add_argument("--split", type=Path,
                    default=Path("model/finetune/cache/split_seed42.json"))
    ap.add_argument("--out",   type=Path,
                    default=Path("model/finetune/cache/aug_fall_sdp_shifted.npz"))
    ap.add_argument("--shifts", type=int, nargs="+",
                    default=SHIFT_STEPS)
    ap.add_argument("--noise-sigma", type=float, default=NOISE_SIGMA)
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()

    build_shifted_cache(
        cache_path=args.cache,
        split_path=args.split,
        out_path=args.out,
        shift_steps=args.shifts,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )
