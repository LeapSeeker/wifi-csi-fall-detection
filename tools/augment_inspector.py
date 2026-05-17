"""SafeSignal 증강 파라미터 검토용 inspector.

대표 window 1개에 대해 4개 증강(jittering / scaling / time_warping / noise_scale)을
적용해 히트맵과 분포 통계를 저장한다. 학습용 데이터셋 증강/저장 도구가 아니라,
파라미터 강도를 사전 점검하기 위한 시각화 전용 스크립트.

실제 전체 증강 적용은 이후 fine-tuning 학습 파이프라인에서 train split에만 수행한다.

CLI:
    python tools/augment_inspector.py <csv_path_or_dir> [--out <output_dir>] [--seed 42]
    python tools/augment_inspector.py --synthetic [--out <output_dir>] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 프로젝트 루트(이 파일의 부모의 부모)를 sys.path 에 추가.
# tools/ 또는 임의 cwd 에서 실행해도 model.* import 가 되도록 한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
except ImportError:
    print(
        "[error] numpy 미설치. `pip install numpy` 후 다시 실행하세요.",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "[error] matplotlib 미설치. `pip install matplotlib` 후 다시 실행하세요.",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import pandas as pd  # synthetic CSV 생성용
except ImportError:
    print(
        "[error] pandas 미설치. `pip install pandas` 후 다시 실행하세요.",
        file=sys.stderr,
    )
    sys.exit(1)

from model.augment.augment import jittering, noise_scale, scaling, time_warping
from model.preprocessing.pipeline import preprocess_safesignal_file_full


DEFAULT_OUT = PROJECT_ROOT / "tools" / "inspect_output"
SAFESIGNAL_N_SC = 52  # LLTF (D-007)
SYNTHETIC_FILENAME = "E1_S01_A_STAND_T001.csv"
SYNTHETIC_N_PACKETS = 300
SYNTHETIC_DT_US = 10_000  # 100Hz


# ── synthetic CSV 생성 ────────────────────────────────────────────────────

def make_synthetic_csv(out_dir: Path, seed: int) -> Path:
    """RPCA degenerate 방어를 통과할 수 있는 합성 SafeSignal CSV 생성."""
    rng = np.random.default_rng(seed)
    n = SYNTHETIC_N_PACKETS

    ts = np.arange(n, dtype=np.int64) * SYNTHETIC_DT_US
    seq = np.arange(n, dtype=np.int64)

    # 정현파 + 노이즈 (constant 입력은 RPCA 방어에 걸리므로 변화량 부여)
    t = np.arange(n, dtype=np.float32) / 100.0  # 초 단위
    # 서브캐리어별 위상 offset, rx별 약간 다른 진폭 baseline
    sc_phase_rx1 = rng.uniform(0, 2 * np.pi, size=SAFESIGNAL_N_SC).astype(np.float32)
    sc_phase_rx2 = rng.uniform(0, 2 * np.pi, size=SAFESIGNAL_N_SC).astype(np.float32)
    base_rx1 = 30.0 + 5.0 * rng.uniform(0, 1, size=SAFESIGNAL_N_SC).astype(np.float32)
    base_rx2 = 28.0 + 5.0 * rng.uniform(0, 1, size=SAFESIGNAL_N_SC).astype(np.float32)

    amp_rx1 = (
        base_rx1[None, :]
        + 2.0 * np.sin(2 * np.pi * 1.5 * t[:, None] + sc_phase_rx1[None, :])
        + rng.normal(0.0, 0.3, size=(n, SAFESIGNAL_N_SC)).astype(np.float32)
    ).astype(np.float32)
    amp_rx2 = (
        base_rx2[None, :]
        + 2.0 * np.sin(2 * np.pi * 1.7 * t[:, None] + sc_phase_rx2[None, :])
        + rng.normal(0.0, 0.3, size=(n, SAFESIGNAL_N_SC)).astype(np.float32)
    ).astype(np.float32)

    data = {"timestamp_us": ts, "seq_rx1": seq, "seq_rx2": seq}
    for i in range(SAFESIGNAL_N_SC):
        data[f"amp_rx1_{i}"] = amp_rx1[:, i]
    for i in range(SAFESIGNAL_N_SC):
        data[f"amp_rx2_{i}"] = amp_rx2[:, i]

    df = pd.DataFrame(data)
    # 컬럼 순서 보장: timestamp_us, seq_rx1, seq_rx2, amp_rx1_*, amp_rx2_*
    ordered = (
        ["timestamp_us", "seq_rx1", "seq_rx2"]
        + [f"amp_rx1_{i}" for i in range(SAFESIGNAL_N_SC)]
        + [f"amp_rx2_{i}" for i in range(SAFESIGNAL_N_SC)]
    )
    df = df[ordered]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / SYNTHETIC_FILENAME
    df.to_csv(csv_path, index=False)
    return csv_path


# ── 입력 해석 ────────────────────────────────────────────────────────────

def resolve_csv_path(arg: str) -> Path:
    """파일 경로면 그대로, 디렉터리면 첫 번째 *.csv (정렬 순)."""
    p = Path(arg)
    if not p.exists():
        raise FileNotFoundError(f"경로가 존재하지 않음: {p}")
    if p.is_file():
        return p
    csvs = sorted(p.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"디렉터리에 *.csv 파일 없음: {p}")
    return csvs[0]


# ── 출력물 생성 ──────────────────────────────────────────────────────────

def save_sdp_distribution(all_values: np.ndarray, out_path: Path) -> dict[str, float]:
    stats = {
        "mean": float(all_values.mean()),
        "std": float(all_values.std()),
        "min": float(all_values.min()),
        "max": float(all_values.max()),
        "p5": float(np.percentile(all_values, 5)),
        "p95": float(np.percentile(all_values, 95)),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_values.ravel(), bins=80, color="#4a7ab8", edgecolor="white")
    ax.set_xlabel("z-score SDP value")
    ax.set_ylabel("frequency")
    ax.set_title("SDP value distribution (all windows, after global z-score)")
    txt = (
        f"mean={stats['mean']:.3f}  std={stats['std']:.3f}\n"
        f"min={stats['min']:.3f}  max={stats['max']:.3f}\n"
        f"p5={stats['p5']:.3f}  p95={stats['p95']:.3f}"
    )
    ax.text(
        0.98, 0.97, txt,
        ha="right", va="top", transform=ax.transAxes,
        fontsize=9, family="monospace",
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return stats


def save_augmentation_heatmaps(
    original: np.ndarray,
    augmented: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """원본 + 4기법 = 5개 heatmap. 컬러바 범위는 원본 기준 고정."""
    vmin = float(original[0].min())
    vmax = float(original[0].max())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    panels = [
        ("Original", original[0]),
        ("Jittering(sigma=0.05)", augmented["jittering"][0]),
        ("Scaling(0.8~1.2)", augmented["scaling"][0]),
        ("TimeWarping", augmented["time_warping"][0]),
        ("NoiseScale", augmented["noise_scale"][0]),
    ]
    last_im = None
    for ax, (title, mat) in zip(axes.flat, panels):
        last_im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("lag (0~19)")
        ax.set_ylabel("time step (0~27)")
    # 남는 마지막 subplot 비활성
    axes.flat[5].axis("off")

    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    fig.suptitle(
        f"SDP heatmaps (vmin={vmin:.3f}, vmax={vmax:.3f}, fixed by Original)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_diff_heatmaps(
    original: np.ndarray,
    augmented: dict[str, np.ndarray],
    out_path: Path,
) -> dict[str, dict[str, float]]:
    """augmented - original diff. 4기법 × 2x2. 대칭 컬러바."""
    keys = ["jittering", "scaling", "time_warping", "noise_scale"]
    titles = {
        "jittering": "Jittering",
        "scaling": "Scaling",
        "time_warping": "TimeWarping",
        "noise_scale": "NoiseScale",
    }
    diffs = {k: augmented[k][0] - original[0] for k in keys}
    vmax = max(float(np.abs(d).max()) for d in diffs.values())
    if vmax == 0.0:  # 모든 기법이 변화 0인 극단 케이스 방어
        vmax = 1e-6
    vmin = -vmax

    stats: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    last_im = None
    for ax, key in zip(axes.flat, keys):
        d = diffs[key]
        max_abs = float(np.abs(d).max())
        mean_abs = float(np.abs(d).mean())
        stats[key] = {"max_abs": max_abs, "mean_abs": mean_abs}
        last_im = ax.imshow(
            d,
            aspect="auto",
            origin="lower",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(
            f"{titles[key]}  max|Δ|={max_abs:.3f}  mean|Δ|={mean_abs:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("lag (0~19)")
        ax.set_ylabel("time step (0~27)")

    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    fig.suptitle(
        f"Augmentation diff (augmented - original), symmetric vmax={vmax:.3f}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return stats


# ── stats summary 텍스트 ─────────────────────────────────────────────────

def write_stats_summary(
    out_path: Path,
    *,
    csv_path: Path,
    synthetic: bool,
    seed: int,
    n_windows: int,
    resample_meta: dict | None,
    sdp_stats: dict[str, float],
    original: np.ndarray,
    augmented: dict[str, np.ndarray],
    diff_stats: dict[str, dict[str, float]],
) -> None:
    orig_std = float(original.std())
    jitter_sigma = 0.05
    sigma_pct = (jitter_sigma / orig_std * 100.0) if orig_std > 0 else float("inf")

    lines: list[str] = []
    lines.append("# SafeSignal augment_inspector — stats summary")
    lines.append("")
    lines.append(f"input_csv         : {csv_path.resolve()}")
    lines.append(f"synthetic         : {synthetic}")
    lines.append(f"seed              : {seed}")
    lines.append(f"total_windows     : {n_windows}")
    if resample_meta is not None:
        lines.append("")
        lines.append("[resample metadata]")
        lines.append(f"  original_rate_hz : {resample_meta['original_rate_hz']}")
        lines.append(f"  gap_count        : {resample_meta['gap_count']}")
        lines.append(f"  max_gap_us       : {resample_meta['max_gap_us']}")

    lines.append("")
    lines.append("[SDP distribution — all windows after global z-score]")
    lines.append(f"  mean : {sdp_stats['mean']:.6f}")
    lines.append(f"  std  : {sdp_stats['std']:.6f}")
    lines.append(f"  min  : {sdp_stats['min']:.6f}")
    lines.append(f"  max  : {sdp_stats['max']:.6f}")
    lines.append(f"  p5   : {sdp_stats['p5']:.6f}")
    lines.append(f"  p95  : {sdp_stats['p95']:.6f}")

    lines.append("")
    lines.append("[Per-technique delta — sample window 0]")
    lines.append(f"  original.std = {orig_std:.6f}")
    for key in ("jittering", "scaling", "time_warping", "noise_scale"):
        aug = augmented[key]
        aug_std = float(aug.std())
        std_change_pct = (
            (aug_std / orig_std - 1.0) * 100.0 if orig_std > 0 else float("nan")
        )
        s = diff_stats[key]
        lines.append(
            f"  {key:14s}: max|Δ|={s['max_abs']:.6f}  "
            f"mean|Δ|={s['mean_abs']:.6f}  "
            f"aug.std={aug_std:.6f}  "
            f"std_change={std_change_pct:+.2f}%"
        )

    lines.append("")
    lines.append("[Parameter sanity — quick read]")
    lines.append(
        f"  jittering sigma=0.05 vs original SDP std={orig_std:.4f} → "
        f"~{sigma_pct:.1f}% of std"
    )
    if sigma_pct < 5:
        lines.append("    → sigma 비중이 작음. 노이즈 강도 약함.")
    elif sigma_pct < 30:
        lines.append("    → 합리적 범위(원본 std 대비 5~30%).")
    else:
        lines.append("    → sigma 비중이 큼. 학습 시 패턴 손상 가능성 검토 필요.")

    scaling_aug = augmented["scaling"]
    original_range = float(original.max() - original.min())
    scaling_diff = np.abs(scaling_aug - original)
    scaling_diff_max = float(scaling_diff.max())
    scaling_diff_mean = float(scaling_diff.mean())
    scaling_aug_std = float(scaling_aug.std())
    scaling_std_change_pct = (
        (scaling_aug_std / orig_std - 1.0) * 100.0 if orig_std > 0 else float("nan")
    )

    lines.append("  scaling range 0.8~1.2:")
    if original_range <= 1e-8:
        lines.append(f"    original_range={original_range:.6f}")
        lines.append(
            "    → 원본 값 범위가 거의 0이므로 scaling 판단 불가. "
            "입력/전처리 상태 확인 필요."
        )
    else:
        ratio = scaling_diff_max / original_range
        lines.append(f"    original_range={original_range:.6f}")
        lines.append(f"    max|Δ|/range={ratio * 100.0:.2f}%")
        lines.append(f"    mean|Δ|={scaling_diff_mean:.6f}")
        lines.append(f"    std_change={scaling_std_change_pct:+.2f}%")
        if ratio < 0.10:
            verdict = (
                "scaling 영향이 약함. 다양성 증가 효과가 제한적일 수 있음."
            )
        elif ratio <= 0.35:
            verdict = (
                "scaling 영향이 원본 범위 대비 완만함. "
                "inspector 기준으로는 무리 없는 범위."
            )
        else:
            verdict = (
                "scaling 영향이 큼. 비낙상 패턴 왜곡 및 FAR 증가 가능성을 "
                "실제 검증에서 확인 필요."
            )
        lines.append(f"    → {verdict}")
    lines.append("")
    lines.append(
        "주의: 위의 적정성 판단은 빠른 sanity check이다. 최종 적정성은 실제 "
        "train split 성능과 FAR로 확정해야 한다."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "SafeSignal 증강 파라미터 검토용 시각화 inspector "
            "(전체 데이터셋 증강/저장 도구 아님)."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="SafeSignal CSV 파일 경로 또는 디렉터리. --synthetic 사용 시 생략 가능.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"출력 디렉터리 (기본: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성 seed (기본: 42)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="합성 CSV를 출력 디렉터리에 생성하여 사용한다.",
    )
    args = parser.parse_args(argv)

    if not args.synthetic and args.csv_path is None:
        parser.error("csv_path가 필요하다 (또는 --synthetic 지정).")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 입력 CSV 결정
    if args.synthetic:
        csv_path = make_synthetic_csv(out_dir, seed=args.seed)
        print(f"[synthetic] 합성 CSV 생성: {csv_path}")
    else:
        try:
            csv_path = resolve_csv_path(args.csv_path)
        except FileNotFoundError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 2

    print(f"[input ] CSV: {csv_path}")
    print(f"[input ] seed: {args.seed}")

    # 2) 전처리
    print("SafeSignal 전처리 시작: RPCA -> ACF -> SDP -> z-score")
    print("RPCA 포함 전처리 중... 시간이 걸릴 수 있음")
    try:
        result = preprocess_safesignal_file_full(csv_path)
    except Exception as e:
        print(f"[error] 전처리 실패: {e!r}", file=sys.stderr)
        return 3

    inputs = result.inputs  # (n_windows, 1, 28, 20)
    if inputs.shape[0] == 0:
        print(
            "[error] 전처리 결과 윈도우 수가 0. 입력 길이/리샘플 metadata 확인.",
            file=sys.stderr,
        )
        return 4

    print(f"[preproc] windows shape: {inputs.shape}")

    # 3) 증강 (개별 함수, 독립 rng)
    seed_seq = np.random.SeedSequence(args.seed)
    jitter_ss, scaling_ss, warp_ss, noise_scale_ss = seed_seq.spawn(4)

    jitter_rng = np.random.default_rng(jitter_ss)
    scaling_rng = np.random.default_rng(scaling_ss)
    warp_rng = np.random.default_rng(warp_ss)
    noise_scale_rng = np.random.default_rng(noise_scale_ss)

    original = inputs[0]  # (1, 28, 20)
    augmented = {
        "jittering": jittering(original, sigma=0.05, rng=jitter_rng),
        "scaling": scaling(original, scale_range=(0.8, 1.2), rng=scaling_rng),
        "time_warping": time_warping(original, rng=warp_rng),
        "noise_scale": noise_scale(
            original, sigma=0.05, scale_range=(0.8, 1.2), rng=noise_scale_rng
        ),
    }

    # 4) 출력물
    dist_path = out_dir / "sdp_distribution.png"
    heatmap_path = out_dir / "augmentation_heatmaps.png"
    diff_path = out_dir / "augmentation_diff_heatmaps.png"
    summary_path = out_dir / "stats_summary.txt"

    sdp_stats = save_sdp_distribution(inputs, dist_path)
    print(f"[saved ] {dist_path}")

    save_augmentation_heatmaps(original, augmented, heatmap_path)
    print(f"[saved ] {heatmap_path}")

    diff_stats = save_diff_heatmaps(original, augmented, diff_path)
    print(f"[saved ] {diff_path}")

    resample_meta = None
    if getattr(result, "resample", None) is not None:
        r = result.resample
        resample_meta = {
            "original_rate_hz": getattr(r, "original_rate_hz", None),
            "gap_count": getattr(r, "gap_count", None),
            "max_gap_us": getattr(r, "max_gap_us", None),
        }

    write_stats_summary(
        summary_path,
        csv_path=csv_path,
        synthetic=args.synthetic,
        seed=args.seed,
        n_windows=int(inputs.shape[0]),
        resample_meta=resample_meta,
        sdp_stats=sdp_stats,
        original=original,
        augmented=augmented,
        diff_stats=diff_stats,
    )
    print(f"[saved ] {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
