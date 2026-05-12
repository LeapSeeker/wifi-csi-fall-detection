"""원본 SDP vs 4기법 증강 결과 heatmap 시각화.

실행:
    python debug/augmentation/check_augment_visual.py          # 합성 데이터
    python debug/augmentation/check_augment_visual.py --show   # 화면 출력
    python debug/augmentation/check_augment_visual.py --npz path/to/sdp.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib이 설치되지 않았습니다. pip install matplotlib")
    sys.exit(1)

from model.augment.augment import jittering, scaling, time_warping, noise_scale


OUTPUT_DIR = Path(__file__).resolve().parent


def load_sdp(npz_path: str | None) -> np.ndarray:
    if npz_path:
        data = np.load(npz_path)
        key = list(data.keys())[0]
        arr = data[key]
        if arr.ndim == 2:
            return arr[None, ...].astype(np.float32)
        return arr.astype(np.float32)
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, (1, 28, 20)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="증강 heatmap 시각화")
    parser.add_argument("--show", action="store_true", help="화면에 표시")
    parser.add_argument("--npz", type=str, default=None, help="실제 SDP npz 경로")
    args = parser.parse_args()

    x = load_sdp(args.npz)
    rng = np.random.default_rng(0)

    fns = [jittering, scaling, time_warping, noise_scale]
    names = ["jittering", "scaling", "time_warping", "noise_scale"]
    augs = [fn(x, rng=np.random.default_rng(i)) for i, fn in enumerate(fns)]

    # 원본 + 4기법 heatmap
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("원본 vs 4기법 증강 SDP Heatmap", fontsize=13)

    all_data = [x] + augs
    all_names = ["원본"] + names
    vmin = min(d[0].min() for d in all_data)
    vmax = max(d[0].max() for d in all_data)

    for ax, data, name in zip(axes, all_data, all_names):
        im = ax.imshow(data[0], aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("lag (0~19)")
        ax.set_ylabel("time step (0~27)")
        diff = float(np.abs(data[0] - x[0]).mean()) if name != "원본" else 0.0
        ax.text(0.5, -0.18, f"mean|Δ|={diff:.4f}", transform=ax.transAxes,
                ha="center", fontsize=9, color="gray")

    plt.colorbar(im, ax=axes[-1])
    plt.tight_layout()
    out1 = OUTPUT_DIR / "augment_visual.png"
    plt.savefig(out1, dpi=100, bbox_inches="tight")
    print(f"저장: {out1}")

    # 차이 heatmap
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
    fig2.suptitle("원본 대비 차이(Δ) Heatmap", fontsize=13)

    diffs = [aug[0] - x[0] for aug in augs]
    dabs = max(abs(d).max() for d in diffs)

    for ax, diff, name in zip(axes2, diffs, names):
        ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-dabs, vmax=dabs)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("lag (0~19)")
        ax.set_ylabel("time step (0~27)")
        ax.text(0.5, -0.18, f"mean|Δ|={abs(diff).mean():.4f}",
                transform=ax.transAxes, ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    out2 = OUTPUT_DIR / "augment_diff.png"
    plt.savefig(out2, dpi=100, bbox_inches="tight")
    print(f"저장: {out2}")

    if args.show:
        plt.show()

    print("DONE")


if __name__ == "__main__":
    main()
