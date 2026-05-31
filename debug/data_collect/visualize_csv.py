"""SafeSignal 수집 CSV를 메뉴 기반으로 시각화한다.

CLI에서 확인할 그래프 종류를 고르고 CSV 파일을 선택하면 matplotlib 창으로
확인한다. 수집 중 빠르게 데이터 형태를 보는 용도라서 저장소의 정식 전처리
경로는 재사용하되, UI는 의존성 없는 번호 입력 방식으로 둔다.

실행:
    python debug/data_collect/visualize_csv.py
    python debug/data_collect/visualize_csv.py --dir data/raw
    python debug/data_collect/visualize_csv.py --file data/raw/E2_S02_A_STAND_T001.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from model.preprocessing.loader import load_safesignal_csv
from model.preprocessing.pipeline import window_to_model_input
from model.preprocessing.resample import resample_to_100hz
from model.preprocessing.window import WINDOW_SIZE, sliding_windows


AMP_RX1_COLS = [f"amp_rx1_{i}" for i in range(52)]
AMP_RX2_COLS = [f"amp_rx2_{i}" for i in range(52)]
REQUIRED_COLS = ["timestamp_us", "seq_rx1", "seq_rx2", *AMP_RX1_COLS, *AMP_RX2_COLS]


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            "[ERROR] matplotlib을 불러오지 못했습니다. "
            "`pip install matplotlib` 후 다시 실행하세요.\n"
            f"원인: {exc}"
        ) from exc
    return plt


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"필수 컬럼 누락({len(missing)}개): {', '.join(missing[:5])}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    return df


def _time_axis_seconds(df: pd.DataFrame) -> np.ndarray:
    ts = df["timestamp_us"].to_numpy(dtype=np.float64)
    if ts.size == 0:
        return np.empty((0,), dtype=np.float64)
    return (ts - ts[0]) / 1_000_000.0


def _amp_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rx1 = df[AMP_RX1_COLS].to_numpy(dtype=np.float32)
    rx2 = df[AMP_RX2_COLS].to_numpy(dtype=np.float32)
    return rx1, rx2


def _select_from_numbered(items: list[Path], prompt: str) -> Path | None:
    if not items:
        print("[INFO] 선택할 CSV 파일이 없습니다.")
        return None
    while True:
        raw = input(prompt).strip()
        if raw.lower() in {"q", "quit", "exit", "0"}:
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(items):
                return items[idx - 1]
        print("번호를 다시 입력하세요. 취소하려면 0 또는 q를 입력하세요.")


def _select_many_from_numbered(items: list[Path], prompt: str) -> list[Path]:
    if not items:
        print("[INFO] 선택할 CSV 파일이 없습니다.")
        return []
    while True:
        raw = input(prompt).strip()
        if raw.lower() in {"q", "quit", "exit", "0"}:
            return []
        selected: list[Path] = []
        ok = True
        for token in raw.replace(",", " ").split():
            if not token.isdigit():
                ok = False
                break
            idx = int(token)
            if not (1 <= idx <= len(items)):
                ok = False
                break
            selected.append(items[idx - 1])
        if ok and selected:
            return selected
        print("번호를 다시 입력하세요. 여러 개는 공백 또는 쉼표로 구분합니다.")


def _choose_file(data_dir: Path, fixed_file: Path | None = None) -> Path | None:
    if fixed_file is not None:
        return fixed_file

    csvs = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        print(f"[INFO] CSV 파일 없음: {data_dir}")
        return None

    keyword = input("파일 필터(엔터=최근 파일 목록, 예: E2 FALL STAND): ").strip()
    if keyword:
        tokens = [t.upper() for t in keyword.split()]
        csvs = [p for p in csvs if all(t in p.name.upper() for t in tokens)]
        if not csvs:
            print("[INFO] 필터와 일치하는 파일이 없습니다.")
            return None

    display = csvs[:50]
    print("\n=== CSV 파일 선택 ===")
    for i, path in enumerate(display, 1):
        print(f"{i:2d}. {path.name}")
    if len(csvs) > len(display):
        print(f"... {len(csvs) - len(display)}개 더 있음. 필터를 더 좁혀주세요.")
    return _select_from_numbered(display, "\n파일 번호: ")


def _choose_compare_files(data_dir: Path, base_path: Path) -> list[Path]:
    csvs = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    csvs = [p for p in csvs if p.resolve() != base_path.resolve()]
    if not csvs:
        print("[INFO] 비교할 다른 CSV 파일이 없습니다.")
        return []

    keyword = input("비교 파일 필터(예: WALK STAND, 엔터=최근 파일): ").strip()
    if keyword:
        tokens = [t.upper() for t in keyword.split()]
        csvs = [p for p in csvs if all(t in p.name.upper() for t in tokens)]
        if not csvs:
            print("[INFO] 필터와 일치하는 비교 파일이 없습니다.")
            return []

    display = csvs[:50]
    print("\n=== 비교 CSV 선택 ===")
    print(f"기준: {base_path.name}")
    for i, path in enumerate(display, 1):
        print(f"{i:2d}. {path.name}")
    if len(csvs) > len(display):
        print(f"... {len(csvs) - len(display)}개 더 있음. 필터를 더 좁혀주세요.")
    return _select_many_from_numbered(display, "\n비교 파일 번호(여러 개 가능): ")


def _print_summary(path: Path, df: pd.DataFrame) -> None:
    rx1, rx2 = _amp_matrices(df)
    seconds = _time_axis_seconds(df)
    duration = float(seconds[-1]) if seconds.size else 0.0
    gaps_signed = np.diff(df["timestamp_us"].to_numpy(dtype=np.float64))
    gaps_abs = np.abs(gaps_signed)
    reversal_count = int(np.sum(gaps_signed < 0))

    print("\n=== CSV 요약 ===")
    print(f"file       : {path.name}")
    print(f"rows       : {len(df)}")
    print(f"duration   : {duration:.3f}s")
    if duration > 0 and len(df) > 1:
        print(f"rate       : {(len(df) - 1) / duration:.2f}Hz")
    print(f"rx1 mean   : {float(rx1.mean()):.3f}  std={float(rx1.std()):.3f}")
    print(f"rx2 mean   : {float(rx2.mean()):.3f}  std={float(rx2.std()):.3f}")
    if gaps_abs.size:
        print(
            "ts gap(abs): "
            f"p95={np.percentile(gaps_abs, 95) / 1000.0:.1f}ms  "
            f"max={gaps_abs.max() / 1000.0:.1f}ms"
        )
        print(
            "ts order   : "
            f"reversals={reversal_count}  "
            f"min_signed_gap={gaps_signed.min() / 1000.0:.1f}ms"
        )
    if "pair_dt_us" in df.columns:
        pair_dt = df["pair_dt_us"].dropna().to_numpy(dtype=np.float64)
        if pair_dt.size:
            print(
                "pair_dt    : "
                f"p50={np.percentile(pair_dt, 50) / 1000.0:.1f}ms  "
                f"p95={np.percentile(pair_dt, 95) / 1000.0:.1f}ms  "
                f"p99={np.percentile(pair_dt, 99) / 1000.0:.1f}ms  "
                f"max={pair_dt.max() / 1000.0:.1f}ms"
            )
    print()


def _show_or_save(fig, save_dir: Path | None, stem: str, show: bool) -> None:
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{stem}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"[saved] {out}")
    if show:
        fig.show()
        input("그래프 창을 확인한 뒤 엔터를 누르세요...")


def plot_mean_timeline(path: Path, df: pd.DataFrame, save_dir: Path | None, show: bool) -> None:
    plt = _import_pyplot()
    t = _time_axis_seconds(df)
    rx1, rx2 = _amp_matrices(df)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, rx1.mean(axis=1), label="Rx1 mean amplitude", linewidth=1.2)
    ax.plot(t, rx2.mean(axis=1), label="Rx2 mean amplitude", linewidth=1.2)
    ax.set_title(path.name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _show_or_save(fig, save_dir, f"{path.stem}_mean_timeline", show)
    plt.close(fig)


def _mean_amplitude_for_mode(df: pd.DataFrame, mode: str) -> np.ndarray:
    rx1, rx2 = _amp_matrices(df)
    if mode == "rx1":
        return rx1.mean(axis=1)
    if mode == "rx2":
        return rx2.mean(axis=1)
    return np.concatenate([rx1, rx2], axis=1).mean(axis=1)


def _amplitude_for_mode(df: pd.DataFrame, mode: str) -> np.ndarray:
    rx1, rx2 = _amp_matrices(df)
    if mode == "rx1":
        return rx1
    if mode == "rx2":
        return rx2
    return np.concatenate([rx1, rx2], axis=1)


def _rolling_std(y: np.ndarray, window: int) -> np.ndarray:
    if y.size == 0:
        return y
    window = max(2, min(window, y.size))
    kernel = np.ones(window, dtype=np.float64) / window
    mean = np.convolve(y, kernel, mode="same")
    mean_sq = np.convolve(y * y, kernel, mode="same")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var)


def _comparison_series(df: pd.DataFrame, mode: str, metric: str) -> np.ndarray:
    amp = _amplitude_for_mode(df, mode)
    y = amp.mean(axis=1)
    if metric == "zmean":
        return (y - y.mean()) / (y.std() + 1e-6)
    if metric == "delta":
        delta = np.abs(np.diff(amp, axis=0)).mean(axis=1)
        if delta.size == 0:
            return delta
        return np.concatenate([[delta[0]], delta])
    if metric == "delta_z":
        delta = np.abs(np.diff(amp, axis=0)).mean(axis=1)
        if delta.size == 0:
            return delta
        delta = np.concatenate([[delta[0]], delta])
        return (delta - delta.mean()) / (delta.std() + 1e-6)
    if metric == "rolling_std":
        return _rolling_std(y.astype(np.float64), window=25)
    if metric == "rolling_std_z":
        std = _rolling_std(y.astype(np.float64), window=25)
        return (std - std.mean()) / (std.std() + 1e-6)
    return y


def _choose_comparison_metric() -> tuple[str, str]:
    options = [
        ("raw", "mean amplitude (raw)"),
        ("zmean", "per-file z-score mean amplitude"),
        ("delta", "mean frame delta |diff|"),
        ("delta_z", "per-file z-score frame delta"),
        ("rolling_std", "rolling std(25 samples)"),
        ("rolling_std_z", "per-file z-score rolling std"),
    ]
    print("\n=== 비교 지표 선택 ===")
    for i, (_, label) in enumerate(options, 1):
        print(f"{i}. {label}")
    raw = input("지표 번호(엔터=4): ").strip()
    idx = int(raw) if raw else 4
    idx = max(1, min(len(options), idx))
    return options[idx - 1]


def plot_compare_mean_timeline(
    path: Path,
    df: pd.DataFrame,
    save_dir: Path | None,
    show: bool,
) -> None:
    plt = _import_pyplot()
    compare_paths = _choose_compare_files(path.parent, path)
    if not compare_paths:
        return

    raw_mode = input("비교 대상(엔터=both, rx1, rx2): ").strip().lower()
    mode = raw_mode if raw_mode in {"rx1", "rx2", "both"} else "both"
    metric, metric_label = _choose_comparison_metric()
    smooth_raw = input("이동평균 window samples(엔터=1, 예: 10): ").strip()
    smooth_window = int(smooth_raw) if smooth_raw else 1
    smooth_window = max(1, smooth_window)

    series: list[tuple[Path, np.ndarray, np.ndarray]] = []
    for item_path, item_df in [(path, df), *[(p, _read_csv(p)) for p in compare_paths]]:
        t = _time_axis_seconds(item_df)
        y = _comparison_series(item_df, mode, metric)
        if smooth_window > 1 and y.size >= smooth_window:
            kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
            y = np.convolve(y, kernel, mode="same")
        series.append((item_path, t, y))

    fig, ax = plt.subplots(figsize=(12, 6))
    for item_path, t, y in series:
        label = item_path.stem
        ax.plot(t, y, linewidth=1.15, label=label)

    ax.set_title(f"{metric_label} comparison ({mode})")
    ax.set_xlabel("time from file start (s)")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    stem = f"{path.stem}_compare_{mode}_{metric}"
    _show_or_save(fig, save_dir, stem, show)
    plt.close(fig)


def plot_heatmap(path: Path, df: pd.DataFrame, save_dir: Path | None, show: bool) -> None:
    plt = _import_pyplot()
    rx1, rx2 = _amp_matrices(df)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, data, title in (
        (axes[0], rx1, "Rx1 amplitude heatmap"),
        (axes[1], rx2, "Rx2 amplitude heatmap"),
    ):
        im = ax.imshow(data.T, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.set_ylabel("subcarrier")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    axes[-1].set_xlabel("packet index")
    fig.suptitle(path.name)
    fig.tight_layout()
    _show_or_save(fig, save_dir, f"{path.stem}_amp_heatmap", show)
    plt.close(fig)


def plot_subcarrier_lines(path: Path, df: pd.DataFrame, save_dir: Path | None, show: bool) -> None:
    plt = _import_pyplot()
    t = _time_axis_seconds(df)
    rx1, rx2 = _amp_matrices(df)
    raw = input("서브캐리어 번호(0~51, 엔터=대표 0/13/26/39/51): ").strip()
    if raw:
        indices = [max(0, min(51, int(raw)))]
    else:
        indices = [0, 13, 26, 39, 51]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, data, title in ((axes[0], rx1, "Rx1"), (axes[1], rx2, "Rx2")):
        for idx in indices:
            ax.plot(t, data[:, idx], linewidth=0.9, label=f"sc{idx}")
        ax.set_title(title)
        ax.set_ylabel("amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=min(len(indices), 5))
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(path.name)
    fig.tight_layout()
    suffix = "sc_" + "_".join(str(i) for i in indices)
    _show_or_save(fig, save_dir, f"{path.stem}_{suffix}", show)
    plt.close(fig)


def plot_pair_quality(path: Path, df: pd.DataFrame, save_dir: Path | None, show: bool) -> None:
    plt = _import_pyplot()
    t = _time_axis_seconds(df)
    gaps_signed_ms = np.diff(df["timestamp_us"].to_numpy(dtype=np.float64)) / 1000.0
    gaps_abs_ms = np.abs(gaps_signed_ms)
    reversal_count = int(np.sum(gaps_signed_ms < 0))

    has_pair = "pair_dt_us" in df.columns
    nrows = 3 if has_pair else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 3.5 * nrows), sharex=False)
    if nrows == 1:
        axes = [axes]

    axes[0].plot(t[1:], gaps_signed_ms, linewidth=0.9)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axes[0].set_title(f"signed timestamp gap (negative = row timestamp reversal, count={reversal_count})")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("signed gap (ms)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t[1:], gaps_abs_ms, linewidth=0.9)
    axes[1].set_title("absolute timestamp gap")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("abs gap (ms)")
    axes[1].grid(True, alpha=0.25)

    if has_pair:
        pair_ms = df["pair_dt_us"].to_numpy(dtype=np.float64) / 1000.0
        axes[2].plot(t, pair_ms, linewidth=0.9)
        axes[2].set_title("Rx1/Rx2 pair delay")
        axes[2].set_xlabel("time (s)")
        axes[2].set_ylabel("pair_dt (ms)")
        axes[2].grid(True, alpha=0.25)

    fig.suptitle(path.name)
    fig.tight_layout()
    _show_or_save(fig, save_dir, f"{path.stem}_pair_quality", show)
    plt.close(fig)


def plot_resampled(path: Path, df: pd.DataFrame, save_dir: Path | None, show: bool) -> None:
    plt = _import_pyplot()
    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)

    if res.resampled_count == 0:
        print("[WARN] 리샘플 결과가 비어 있습니다.")
        return

    t_raw = (raw.timestamps_us - raw.timestamps_us[0]) / 1_000_000.0
    t_res = (res.timestamps_us - res.timestamps_us[0]) / 1_000_000.0
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    axes[0].plot(t_raw, raw.amplitude.mean(axis=1), label="raw", linewidth=1.0)
    axes[0].plot(t_res, res.amplitude.mean(axis=1), label="resampled 100Hz", linewidth=1.0)
    axes[0].set_title(
        f"mean amplitude: raw {res.original_count} rows -> {res.resampled_count} rows, "
        f"rate={res.original_rate_hz:.2f}Hz"
    )
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("mean amplitude")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    im1 = axes[1].imshow(raw.amplitude.T, aspect="auto", origin="lower", interpolation="nearest")
    axes[1].set_title("raw both-Rx heatmap")
    axes[1].set_ylabel("subcarrier concat")
    fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.01)

    im2 = axes[2].imshow(res.amplitude.T, aspect="auto", origin="lower", interpolation="nearest")
    axes[2].set_title("resampled both-Rx heatmap")
    axes[2].set_xlabel("100Hz sample index")
    axes[2].set_ylabel("subcarrier concat")
    fig.colorbar(im2, ax=axes[2], fraction=0.025, pad=0.01)

    fig.suptitle(path.name)
    fig.tight_layout()
    _show_or_save(fig, save_dir, f"{path.stem}_resampled", show)
    plt.close(fig)


def plot_sdp(path: Path, df: pd.DataFrame, save_dir: Path | None, show: bool) -> None:
    plt = _import_pyplot()
    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    windows = sliding_windows(
        res.amplitude,
        window_size=WINDOW_SIZE,
        stride=100,
        drop_last=True,
    )
    if windows.shape[0] == 0:
        print("[WARN] 300 frame window를 만들 수 없습니다.")
        return

    print(f"윈도우 개수: {windows.shape[0]} (stride=100)")
    raw_idx = input("볼 window index(엔터=0): ").strip()
    idx = int(raw_idx) if raw_idx else 0
    idx = max(0, min(windows.shape[0] - 1, idx))
    raw_iter = input("RPCA max_iter(엔터=80, 정식 분석은 200): ").strip()
    rpca_max_iter = int(raw_iter) if raw_iter else 80

    model_input = window_to_model_input(windows[idx], rpca_max_iter=rpca_max_iter)
    sdp = model_input[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(sdp, aspect="auto", origin="lower", cmap="RdBu_r")
    ax.set_title(f"{path.name} | SDP z-score | window={idx} | rpca_iter={rpca_max_iter}")
    ax.set_xlabel("ACF lag")
    ax.set_ylabel("subwindow")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _show_or_save(fig, save_dir, f"{path.stem}_sdp_w{idx}", show)
    plt.close(fig)


MENU = [
    ("CSV 요약 출력", _print_summary),
    ("Rx1/Rx2 평균 진폭 타임라인", plot_mean_timeline),
    ("여러 CSV 평균 진폭 비교", plot_compare_mean_timeline),
    ("Rx1/Rx2 진폭 heatmap", plot_heatmap),
    ("서브캐리어별 진폭 라인", plot_subcarrier_lines),
    ("timestamp gap / pair delay 그래프", plot_pair_quality),
    ("100Hz 리샘플 전후 비교", plot_resampled),
    ("전처리 후 SDP heatmap", plot_sdp),
]


def _choose_action() -> int | None:
    print("\n=== SafeSignal CSV 시각화 메뉴 ===")
    for i, (label, _) in enumerate(MENU, 1):
        print(f"{i}. {label}")
    print("0. 종료")
    while True:
        raw = input("확인할 동작 번호: ").strip()
        if raw in {"0", "q", "Q"}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(MENU):
            return int(raw) - 1
        print("번호를 다시 입력하세요.")


def run_interactive(args: argparse.Namespace) -> int:
    data_dir = (PROJECT_ROOT / args.dir).resolve()
    fixed_file = (PROJECT_ROOT / args.file).resolve() if args.file else None
    save_dir = (PROJECT_ROOT / args.save_dir).resolve() if args.save_dir else None

    while True:
        action_idx = _choose_action()
        if action_idx is None:
            return 0

        path = _choose_file(data_dir, fixed_file)
        if path is None:
            if fixed_file is not None:
                return 1
            continue

        try:
            df = _read_csv(path)
            label, func = MENU[action_idx]
            print(f"\n[run] {label}: {path.name}")
            if func is _print_summary:
                func(path, df)
            else:
                func(path, df, save_dir, not args.no_show)
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")

        if fixed_file is not None and args.once:
            return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeSignal CSV 메뉴 기반 시각화")
    parser.add_argument("--dir", default="data/raw", help="CSV 폴더")
    parser.add_argument("--file", default=None, help="CSV 파일 직접 지정")
    parser.add_argument("--save-dir", default=None, help="그래프 PNG 저장 폴더")
    parser.add_argument("--no-show", action="store_true", help="창을 띄우지 않고 저장만 수행")
    parser.add_argument(
        "--once",
        action="store_true",
        help="--file 사용 시 동작 1회 실행 후 종료",
    )
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run_interactive(parse_args()))


if __name__ == "__main__":
    main()
