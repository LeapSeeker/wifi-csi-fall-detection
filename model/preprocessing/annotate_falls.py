"""낙상 onset 자동 탐지 + 이상치 수동 검토.

알고리즘:
  100Hz 리샘플 → 서브캐리어 평균 에너지 → rolling std → 배경 대비
  급격히 상승하는 첫 지점 = 낙상 시작 패킷

사용법:
    # 1단계: 자동 탐지 (전체 낙상 파일)
    python -m model.preprocessing.annotate_falls \\
        --src dummy_src --out dummy_src/fall_onsets.json

    # 2단계: 이상치 시각 검토 (선택)
    python -m model.preprocessing.annotate_falls \\
        --src dummy_src --out dummy_src/fall_onsets.json --review
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.preprocessing.loader import load_safesignal_csv, parse_safesignal_filename
from model.preprocessing.resample import resample_to_100hz

FALL_ACTIVITIES = {
    "FALL_SIT_B", "FALL_SIT_F",
    "FALL_STD_B", "FALL_STD_F",
    "FALL_WALK_B", "FALL_WALK_F",
}

# 탐지 범위: 수집 구조상 onset 은 30~220패킷 사이에 존재
SEARCH_MIN = 50   # 0.5s: 너무 이른 감지 차단
SEARCH_MAX = 180  # 1.8s: 이후는 대기/여유 구간으로 판단
ROLLING_W  = 30   # rolling std 윈도우 (더 넓게 → 안정적)
BG_PACKETS = 60   # 배경 레벨 계산용 앞부분 패킷 수
BG_SIGMA   = 3.0  # 배경의 3σ 이상 = 확실한 움직임


def _rolling_std(energy: np.ndarray, w: int) -> np.ndarray:
    out = np.empty_like(energy)
    for i in range(len(energy)):
        lo = max(0, i - w // 2)
        hi = min(len(energy), i + w // 2)
        out[i] = energy[lo:hi].std()
    return out


def detect_onset(amp_100hz: np.ndarray) -> int:
    """100Hz 진폭 배열에서 낙상 onset 패킷 인덱스 반환.

    1) 서브캐리어 분산 합산 → 시간별 활동량 지표
    2) rolling std 로 평활화
    3) 배경(pre-fall 정지 구간)의 3σ 초과 지점 = onset
    4) fallback: 탐색 범위 내 최대 에너지 기울기
    """
    # 각 패킷의 서브캐리어 간 분산 (움직임에 민감)
    per_packet_var = np.var(amp_100hz, axis=1).astype(np.float64)
    rstd = _rolling_std(per_packet_var, ROLLING_W)

    # 배경: SEARCH_MIN 이전 패킷들 (정지 구간)
    bg_region = rstd[:SEARCH_MIN]
    bg_mean = bg_region.mean()
    bg_std  = bg_region.std() + 1e-6
    threshold = bg_mean + BG_SIGMA * bg_std

    search = rstd[SEARCH_MIN:SEARCH_MAX]

    # 임계값을 처음 넘는 지점
    over = np.where(search > threshold)[0]
    if len(over) > 0:
        return int(SEARCH_MIN + over[0])

    # fallback: 탐색 범위 내 최대 기울기
    grad = np.diff(search)
    return int(SEARCH_MIN + int(np.argmax(grad)))


def load_and_resample(path: Path) -> np.ndarray:
    raw = load_safesignal_csv(path, rx="both")
    res = resample_to_100hz(raw.amplitude, raw.timestamps_us)
    return res.amplitude


def run_detection(src: Path) -> dict[str, int]:
    """전체 낙상 파일 자동 탐지. {filename: onset_packet} 반환."""
    fall_files = []
    for p in sorted(src.rglob("*.csv")):
        try:
            meta = parse_safesignal_filename(p)
            if meta.activity.upper() in FALL_ACTIVITIES:
                fall_files.append(p)
        except ValueError:
            continue

    if not fall_files:
        raise FileNotFoundError(f"낙상 CSV 없음: {src}")

    print(f"낙상 파일 {len(fall_files)}개 탐지 중...")
    annotations: dict[str, int] = {}
    errors = []

    for i, p in enumerate(fall_files, 1):
        try:
            amp = load_and_resample(p)
            onset = detect_onset(amp)
            annotations[p.name] = onset
            if i % 50 == 0:
                print(f"  {i}/{len(fall_files)} 완료")
        except Exception as e:
            errors.append((p.name, str(e)))

    if errors:
        print(f"\n오류 {len(errors)}개:")
        for name, err in errors:
            print(f"  {name}: {err}")

    return annotations


def print_summary(annotations: dict[str, int]) -> None:
    onsets = np.array(list(annotations.values()))
    print(f"\n[ 탐지 결과 요약 ]")
    print(f"  총 파일  : {len(onsets)}")
    print(f"  평균 onset: {onsets.mean():.1f} 패킷 ({onsets.mean()/100:.2f}s)")
    print(f"  표준편차  : {onsets.std():.1f}")
    print(f"  최솟값    : {onsets.min()} 패킷")
    print(f"  최댓값    : {onsets.max()} 패킷")

    outliers_low  = {k: v for k, v in annotations.items() if v < 50}
    outliers_high = {k: v for k, v in annotations.items() if v > 180}
    print(f"  이상치(< 50 패킷): {len(outliers_low)}개")
    print(f"  이상치(>180 패킷): {len(outliers_high)}개")

    hist, edges = np.histogram(onsets, bins=range(0, 250, 20))
    print("\n  onset 분포 (20패킷 단위):")
    for cnt, lo in zip(hist, edges):
        bar = "#" * (cnt // 3)
        print(f"    {int(lo):>3}~{int(lo)+20:<3}: {cnt:>3}  {bar}")


def export_review_plots(
    src: Path,
    annotations: dict[str, int],
    out_dir: Path,
    n_extreme: int = 30,
) -> None:
    """onset 값 기준 상하위 n_extreme개 파일을 PNG로 내보내기.

    낮은 onset(너무 이른 탐지) + 높은 onset(너무 늦은 탐지) 각각 확인.
    out_dir에 PNG 저장 → 파일 탐색기에서 검토 후 corrections.json 작성.
    """
    import matplotlib
    matplotlib.use("Agg")   # 화면 없이 파일 저장
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_items = sorted(annotations.items(), key=lambda x: x[1])
    candidates = sorted_items[:n_extreme] + sorted_items[-n_extreme:]
    # 중복 제거
    seen: set[str] = set()
    unique = [(k, v) for k, v in candidates if not (k in seen or seen.add(k))]

    print(f"플롯 생성 중 ({len(unique)}개)...")
    for fname, onset in unique:
        matches = list(src.rglob(fname))
        if not matches:
            continue
        try:
            amp = load_and_resample(matches[0])
        except Exception:
            continue

        energy = np.mean(np.abs(amp), axis=1)
        var    = np.var(amp, axis=1)
        t      = np.arange(len(energy)) / 100.0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax1.plot(t, energy, lw=0.8, color="steelblue")
        ax1.axvline(onset / 100, color="red", lw=2, label=f"onset={onset}pkt ({onset/100:.2f}s)")
        ax1.set_ylabel("mean amplitude")
        ax1.legend(fontsize=9)
        ax1.set_title(fname, fontsize=9)

        ax2.plot(t, var, lw=0.8, color="orange")
        ax2.axvline(onset / 100, color="red", lw=2)
        ax2.set_ylabel("subcarrier variance")
        ax2.set_xlabel("time (s)")

        plt.tight_layout()
        safe_name = fname.replace(".csv", f"_onset{onset}.png")
        fig.savefig(out_dir / safe_name, dpi=80)
        plt.close(fig)

    print(f"저장 완료: {out_dir}")
    print(f"\n검토 방법:")
    print(f"  1. {out_dir} 폴더를 파일 탐색기로 열기")
    print(f"  2. 빨간 선이 실제 낙상 시작과 다른 파일 확인")
    print(f"  3. corrections.json 파일 작성:")
    print(f'     {{"파일명.csv": 실제_onset_패킷, ...}}')
    print(f"  4. python -m model.preprocessing.annotate_falls \\")
    print(f"       --src dummy_src --out dummy_src/fall_onsets.json \\")
    print(f"       --apply-corrections dummy_src/corrections.json")


def apply_corrections(annotations: dict[str, int], corrections_path: Path) -> dict[str, int]:
    """corrections.json을 fall_onsets.json에 반영."""
    import json as _json
    corrections = _json.loads(corrections_path.read_text(encoding="utf-8"))
    updated = dict(annotations)
    for fname, onset in corrections.items():
        if fname in updated:
            print(f"  수정: {fname}  {updated[fname]} -> {onset}")
            updated[fname] = int(onset)
        else:
            print(f"  경고: {fname} 어노테이션에 없음 (스킵)")
    print(f"총 {len(corrections)}개 수정 반영")
    return updated


def review_outliers(src: Path, annotations: dict[str, int]) -> dict[str, int]:
    """이상치 파일을 시각적으로 검토하고 onset 수정."""
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib 사용 불가 - 시각 검토 건너뜀")
        return annotations

    outliers = {k: v for k, v in annotations.items()
                if v < 50 or v > 180}
    if not outliers:
        print("이상치 없음 - 검토 불필요")
        return annotations

    print(f"\n이상치 {len(outliers)}개 검토 (Enter=유지, 숫자=수정, q=종료):")
    updated = dict(annotations)

    for fname, onset in outliers.items():
        # 파일 찾기
        matches = list(src.rglob(fname))
        if not matches:
            continue
        path = matches[0]

        try:
            amp = load_and_resample(path)
        except Exception:
            continue

        energy = np.mean(np.abs(amp), axis=1)
        from model.preprocessing.annotate_falls import _rolling_std
        rstd = _rolling_std(energy, ROLLING_W)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        t = np.arange(len(energy)) / 100.0
        ax1.plot(t, energy, lw=0.8, alpha=0.7)
        ax1.axvline(onset / 100, color="r", lw=2, label=f"onset={onset}")
        ax1.set_ylabel("mean amplitude")
        ax1.legend()
        ax1.set_title(fname)

        ax2.plot(t, rstd, lw=0.8, color="orange")
        ax2.axvline(onset / 100, color="r", lw=2)
        ax2.set_ylabel("rolling std")
        ax2.set_xlabel("time (s)")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        ans = input(f"  {fname} onset={onset} → Enter=유지, 숫자=수정, q=종료: ").strip()
        plt.close(fig)

        if ans.lower() == "q":
            break
        if ans.isdigit():
            new_onset = int(ans)
            updated[fname] = new_onset
            print(f"    수정: {onset} → {new_onset}")

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="낙상 onset 자동 탐지")
    parser.add_argument("--src", type=Path, default=Path("dummy_src"))
    parser.add_argument("--out", type=Path, default=Path("dummy_src/fall_onsets.json"))
    parser.add_argument("--review", action="store_true",
                        help="이상치 시각 검토 활성화")
    parser.add_argument("--export-plots", type=Path, default=None,
                        help="onset 상하위 30개 PNG 내보낼 폴더 (검토용)")
    parser.add_argument("--apply-corrections", type=Path, default=None,
                        help="corrections.json 경로 → fall_onsets.json 수정 반영")
    args = parser.parse_args()

    # 기존 어노테이션 로드 (재실행 시 이어서 작업)
    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"기존 어노테이션 {len(existing)}개 로드")
    else:
        existing = {}

    annotations = run_detection(args.src)
    # 기존 수동 수정 보존 (재탐지해도 덮어쓰지 않음)
    for k, v in existing.items():
        if k in annotations:
            annotations[k] = v  # 기존 값 우선

    # 탐지 결과를 먼저 저장 (이후 단계 실패해도 보존)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(annotations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n저장: {args.out} ({len(annotations)}개)")

    print_summary(annotations)

    if args.export_plots:
        export_review_plots(args.src, annotations, args.export_plots)

    if args.apply_corrections and args.apply_corrections.exists():
        annotations = apply_corrections(annotations, args.apply_corrections)
        args.out.write_text(
            json.dumps(annotations, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"수정 저장: {args.out}")

    if args.review:
        annotations = review_outliers(args.src, annotations)
        args.out.write_text(
            json.dumps(annotations, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"수정 저장: {args.out}")


if __name__ == "__main__":
    main()
