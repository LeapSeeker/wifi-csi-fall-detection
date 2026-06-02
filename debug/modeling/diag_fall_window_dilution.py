"""fall window weak-label transient dilution 진단 (읽기 전용).

배경
----
fall 세션은 event 중심 4초 프로토콜(대기 1s + 낙상 2s + 정지 1s)이고 window는
300-frame(3s)이다. 4s 세션 → 보통 정방향 window[0:300]과 tail window[n-300:n] 2개가
모두 fall 라벨이 된다. 두 window 다 낙상 구간(세션 t=1~3s)을 포함하므로 mislabel은
아니지만, 3s window 안에서 transient(낙상 순간)가 앞뒤 정적에 희석되는 정도가
window/sub-type마다 다르다. 이 스크립트는 sub-type별로 어느 window가 더 event-like
한지 진단한다.

원칙
----
- 읽기 전용. pipeline/cache/모델 파일 수정 금지. 전처리 서브함수만 호출.
- 진단 입력·등가검증은 summary.json의 source_dir(data/cleaned) 기준.
- window 구성 하드코딩 금지 — cache builder와 동일 방식(window_size=300,
  stride=None→300, tail_window=True, pad_short=False)으로 생성하고 실제 start_frame 기록.
- z-score 직전 pre-zscore SDP(28,20)를 진단 지표 산출에 사용.
- peak_time_s 고정 offset 금지: 각 window 실제 start_frame 사용.

산출물은 debug/modeling/diag_out/ (비추적). STATE/markdown 리포트 작성 안 함.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # import 직후 즉시 (headless)
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model.preprocessing.acf import N_LAGS
from model.preprocessing.loader import (
    SAFESIGNAL_N_SUBCARRIERS,
    load_safesignal_csv,
    parse_safesignal_filename,
)
from model.preprocessing.pipeline import (
    preprocess_safesignal_file,
    preprocess_safesignal_file_full,
    window_to_model_input,
)
from model.preprocessing.resample import resample_to_100hz
from model.preprocessing.rpca import DEFAULT_MAX_ITER, rpca_sparse
from model.preprocessing.sdp import SUB_STRIDE, SUB_W, W_T, stacked_doppler_profile
from model.preprocessing.window import WINDOW_SIZE, sliding_windows

# ─── 상수: cache builder 기본값과 일치 (debug/modeling/build_safesignal_cache.py) ──
SOURCE_DIR = PROJECT_ROOT / "data" / "cleaned"
SUMMARY_JSON = PROJECT_ROOT / "model" / "finetune" / "cache" / "safesignal_e1234_finetune7.summary.json"
OUT_DIR = Path(__file__).resolve().parent / "diag_out"

RX = "both"
TARGET_HZ = 100.0
MAX_GAP_MS = 100.0
STRIDE = None          # → window_size (=300), cache builder 기본값
TAIL_WINDOW = True     # cache builder 기본값
PAD_SHORT = False
RPCA_MAX_ITER = DEFAULT_MAX_ITER   # 200
RPCA_TOL = None
ZSCORE_EPS = 1e-6      # pipeline.window_to_model_input 과 동일

SUB_TYPES = ["FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B", "FALL_WALK_F", "FALL_WALK_B"]
FALL_RANGE_S = (1.0, 3.0)   # 세션 프로토콜상 낙상 구간 (대기 1s 이후 2s)
N_PER_SUBTYPE_PER_COMBO = 2  # (env,subject) 조합당 trial 수 → 6 combo × 2 = 12/sub-type
SEED = 42

# 후보 checkpoint (within-subject 우선; GPU full-run 없으면 CPU fallback)
CKPT_CANDIDATES = [
    PROJECT_ROOT / "model" / "finetune" / "checkpoints_compare6_cpu" / "best_operating.pt",
]


# ─── window 생성 (cache builder 와 동일 방식 + 실제 start_frame) ──────────────

def windows_with_starts(amp_100hz: np.ndarray) -> tuple[np.ndarray, list[int], list[str]]:
    """sliding_windows 와 동일 결과 + 각 window 실제 start_frame, kind 반환.

    window.py 로직을 그대로 재현해 start_frame 을 계산하고, sliding_windows
    출력과 allclose 로 자체 검증한다. stride=None→window_size, tail_window=True.
    """
    n = amp_100hz.shape[0]
    win = sliding_windows(
        amp_100hz,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        drop_last=True,
        tail_window=TAIL_WINDOW,
        pad_short=PAD_SHORT,
    )
    # start_frame 재현
    stride = WINDOW_SIZE if STRIDE is None else STRIDE
    starts: list[int] = []
    kinds: list[str] = []
    if n >= WINDOW_SIZE:
        n_fwd = 1 + (n - WINDOW_SIZE) // stride
        fwd_starts = [k * stride for k in range(n_fwd)]
        starts.extend(fwd_starts)
        kinds.extend(["forward"] * n_fwd)
        last_end = fwd_starts[-1] + WINDOW_SIZE
        if TAIL_WINDOW and last_end < n:
            starts.append(n - WINDOW_SIZE)
            kinds.append("tail")
    # 자체 검증: 재현한 start 로 슬라이스한 window 가 sliding_windows 출력과 일치?
    if win.shape[0] != len(starts):
        raise RuntimeError(
            f"window count mismatch: sliding_windows={win.shape[0]} vs reconstructed={len(starts)}"
        )
    for i, s in enumerate(starts):
        if not np.allclose(win[i], amp_100hz[s : s + WINDOW_SIZE]):
            raise RuntimeError(f"window {i} (start={s}) slice mismatch with sliding_windows output")
    return win, starts, kinds


# ─── pre-zscore SDP 서브함수 경로 ────────────────────────────────────────────

def window_pre_zscore_sdp(window: np.ndarray) -> np.ndarray:
    """단일 window (300, n_sc) → z-score 직전 pre-zscore SDP (28, 20).

    pipeline.window_to_model_input 의 z-score 직전까지와 동일한 서브함수 호출.
    """
    sparse = rpca_sparse(window, max_iter=RPCA_MAX_ITER, tol=RPCA_TOL)
    sdp = stacked_doppler_profile(sparse, sub_w=SUB_W, stride=SUB_STRIDE, n_lags=N_LAGS)
    return sdp  # (28, 20), pre-zscore


def apply_global_zscore(sdp: np.ndarray) -> np.ndarray:
    return (sdp - sdp.mean()) / (sdp.std() + ZSCORE_EPS)


# ─── 지표 ────────────────────────────────────────────────────────────────────

def subwindow_energy(sdp: np.ndarray) -> np.ndarray:
    """28개 subwindow 각각의 energy (lag 축 L2^2 합). pre-zscore SDP 기준."""
    return (sdp ** 2).sum(axis=1)  # (28,)


def subwindow_center_time_s(start_frame: int, i: int) -> float:
    """subwindow i 중심의 절대 세션 시간(s). center frame = start + i*stride + sub_w/2."""
    return (start_frame + i * SUB_STRIDE + SUB_W / 2) / 100.0


def expected_fall_subwindows(start_frame: int) -> list[int]:
    """subwindow center 시간이 FALL_RANGE_S 안에 드는 i 목록 (동적)."""
    lo, hi = FALL_RANGE_S
    return [i for i in range(W_T) if lo <= subwindow_center_time_s(start_frame, i) <= hi]


@dataclass
class WindowRecord:
    sub_type: str
    filename: str
    subject: int
    env: int
    trial: int
    window_kind: str
    start_frame: int
    n_frames: int
    sdp_mean_abs: float
    sdp_std: float
    peak_subwindow: int
    peak_time_s: float
    peak_in_fall_range: bool
    peak_energy_ratio: float       # peak subwindow energy / 전체 energy 합 (집중도)
    energy_in_fall_ratio: float    # fall-range subwindow energy 합 / 전체 (event 집중 비율)
    expected_fall_subwindows: list[int]
    attn_peak: int | None
    attn_peak_time_s: float | None
    attn_in_fall_range: bool | None
    checkpoint_used: str | None
    pre_zscore_sdp: np.ndarray     # (28,20) — heatmap 집계용 (CSV 미포함)
    energy: np.ndarray             # (28,)   — 분포 (CSV 미포함)


# ─── 파일 선택 ────────────────────────────────────────────────────────────────

def discover_fall_files() -> dict[str, list[Path]]:
    """sub-type → 파일 목록. side(_S) 제외, summary source_dir 기준."""
    by_subtype: dict[str, list[Path]] = {st: [] for st in SUB_TYPES}
    for path in sorted(SOURCE_DIR.glob("*.csv")):
        try:
            meta = parse_safesignal_filename(path)
        except ValueError:
            continue
        act = meta.activity
        if act in by_subtype:  # FALL_*_S 는 SUB_TYPES 에 없으므로 자동 제외
            by_subtype[act].append(path)
    return by_subtype


def select_samples(by_subtype: dict[str, list[Path]], rng: np.random.Generator) -> dict[str, list[Path]]:
    """sub-type별 (env,subject) 조합당 N개 trial 샘플. S03 자연 포함, 전체 목록 반환."""
    selected: dict[str, list[Path]] = {}
    for st in SUB_TYPES:
        files = by_subtype[st]
        # (env, subject) 조합별 그룹
        groups: dict[tuple[int, int], list[Path]] = {}
        for p in files:
            m = parse_safesignal_filename(p)
            groups.setdefault((m.environment, m.subject), []).append(p)
        picks: list[Path] = []
        for key in sorted(groups):
            pool = sorted(groups[key], key=lambda x: x.name)
            k = min(N_PER_SUBTYPE_PER_COMBO, len(pool))
            idx = rng.choice(len(pool), size=k, replace=False)
            picks.extend([pool[i] for i in sorted(idx)])
        selected[st] = sorted(picks, key=lambda x: x.name)
    return selected


# ─── 등가검증 ────────────────────────────────────────────────────────────────

def equivalence_check(path: Path) -> dict:
    """training 경로(preprocess_safesignal_file_full) vs 서브함수 경로 등가 확인.

    cache row 순서 비의존: 같은 파일을 두 경로로 처리해 (N,1,28,20) 를 만들고
    allclose. 서브함수 경로는 동일 window(sliding_windows) → rpca_sparse →
    stacked_doppler_profile → global z-score.
    """
    # training 경로
    full = preprocess_safesignal_file_full(
        path,
        rx=RX,
        target_hz=TARGET_HZ,
        max_gap_ms=MAX_GAP_MS,
        stride=STRIDE,
        tail_window=TAIL_WINDOW,
        pad_short=PAD_SHORT,
        rpca_max_iter=RPCA_MAX_ITER,
        rpca_tol=RPCA_TOL,
    )
    X_train = full.inputs  # (N,1,28,20) post-zscore

    # 서브함수 경로: 동일 window → pre-zscore SDP → z-score
    pre = preprocess_safesignal_file(
        path,
        rx=RX,
        target_hz=TARGET_HZ,
        max_gap_ms=MAX_GAP_MS,
        stride=STRIDE,
        tail_window=TAIL_WINDOW,
        pad_short=PAD_SHORT,
    )
    sub_out = np.empty_like(X_train)
    for i, w in enumerate(pre.windows):
        sdp = window_pre_zscore_sdp(w)
        sub_out[i] = apply_global_zscore(sdp)[None, ...]

    ok = bool(np.allclose(X_train, sub_out, rtol=1e-4, atol=1e-5)) and X_train.shape == sub_out.shape
    max_abs_diff = float(np.abs(X_train - sub_out).max()) if X_train.shape == sub_out.shape else float("nan")
    return {
        "filename": path.name,
        "shape_train": X_train.shape,
        "shape_sub": sub_out.shape,
        "allclose": ok,
        "max_abs_diff": max_abs_diff,
        "funcs": "preprocess_safesignal_file_full  vs  preprocess_safesignal_file + rpca_sparse + stacked_doppler_profile + global z-score",
    }


# ─── 모델/attention ──────────────────────────────────────────────────────────

def load_model_and_meta():
    """checkpoint 로드. (model, ckpt_meta, ckpt_path) 또는 (None, None, None)."""
    import torch

    from model.pretrained.model import CNNGRUAttention

    searched = []
    for cand in CKPT_CANDIDATES:
        searched.append(str(cand))
        if cand.exists():
            ckpt = torch.load(cand, map_location="cpu", weights_only=False)
            classes = ckpt.get("classes")
            policy = ckpt.get("class_policy")
            n_classes = len(classes) if classes else 6
            model = CNNGRUAttention(n_classes=n_classes)
            model.load_state_dict(ckpt["model"])
            model.eval()
            meta = {
                "path": str(cand),
                "classes": classes,
                "class_policy": policy,
                "split": ckpt.get("args", {}).get("eval_split") if isinstance(ckpt.get("args"), dict) else None,
                "threshold": ckpt.get("threshold"),
                "n_classes": n_classes,
            }
            return model, meta, str(cand)
    print(f"[attn] checkpoint 미발견. 탐색 경로: {searched}")
    return None, None, None


def attention_weights(model, x_zscored: np.ndarray) -> np.ndarray:
    """(N,1,28,20) → attention weights (N,28). forward hook 불필요 — 모델이
    return_attention=True 로 직접 반환(weights 가 T=28 축에 걸림)."""
    import torch

    with torch.no_grad():
        t = torch.from_numpy(x_zscored.astype(np.float32))
        _, weights = model(t, return_attention=True)
    return weights.cpu().numpy()  # (N,28)


# ─── 집계/heatmap ────────────────────────────────────────────────────────────

def save_heatmaps(records: list[WindowRecord]) -> None:
    for st in SUB_TYPES:
        fwd = [r.pre_zscore_sdp for r in records if r.sub_type == st and r.window_kind == "forward"]
        tail = [r.pre_zscore_sdp for r in records if r.sub_type == st and r.window_kind == "tail"]
        if not fwd and not tail:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, data, title in zip(axes, [fwd, tail], ["forward [0:300]", "tail [n-300:n]"]):
            ax.set_title(f"{st} {title} (mean, n={len(data)})")
            if data:
                mean_sdp = np.mean(np.stack(data, axis=0), axis=0)  # (28,20)
                im = ax.imshow(mean_sdp, aspect="auto", origin="lower", cmap="viridis")
                fig.colorbar(im, ax=ax, fraction=0.046)
                ax.set_xlabel("ACF lag (1..20)")
                ax.set_ylabel("subwindow (0..27)")
            else:
                ax.text(0.5, 0.5, "no windows", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"sdp_heatmap_{st}.png", dpi=110)
        plt.close(fig)


def summarize(records: list[WindowRecord]) -> None:
    print("\n" + "=" * 78)
    print("SUB-TYPE 층화 요약: [forward] vs [tail] event-likeness")
    print("=" * 78)
    print("  event-likeness 지표:")
    print("   - peak_in_fall%      : peak subwindow 가 세션 [1,3]s 안에 든 window 비율")
    print("   - energy_in_fall%    : fall-range subwindow energy / 전체 energy (event 집중도)")
    print("   - peak_conc          : peak subwindow energy / 전체 (transient 집중도, 높을수록 event-like)")
    print("   - sdp_std            : pre-zscore SDP 표준편차 (동적 변화량 proxy)")

    def agg(rs: list[WindowRecord]) -> str:
        if not rs:
            return "  (n=0)"
        n = len(rs)
        pif = np.mean([r.peak_in_fall_range for r in rs]) * 100
        eif = np.mean([r.energy_in_fall_ratio for r in rs]) * 100
        conc = np.mean([r.peak_energy_ratio for r in rs])
        std = np.mean([r.sdp_std for r in rs])
        mab = np.mean([r.sdp_mean_abs for r in rs])
        attn = [r for r in rs if r.attn_in_fall_range is not None]
        attn_s = ""
        if attn:
            aif = np.mean([r.attn_in_fall_range for r in attn]) * 100
            attn_s = f"  attn_in_fall%={aif:5.1f}"
        return (f"  n={n:3d}  peak_in_fall%={pif:5.1f}  energy_in_fall%={eif:5.1f}  "
                f"peak_conc={conc:.3f}  sdp_std={std:.4f}  sdp_mean_abs={mab:.4f}{attn_s}")

    families = {"SIT/STD (transient 희석 심함)": ["FALL_SIT_F", "FALL_SIT_B", "FALL_STD_F", "FALL_STD_B"],
                "WALK (이동 baseline)": ["FALL_WALK_F", "FALL_WALK_B"]}

    for st in SUB_TYPES:
        rs = [r for r in records if r.sub_type == st]
        print(f"\n[{st}]")
        print(f"  forward: {agg([r for r in rs if r.window_kind == 'forward'])}")
        print(f"  tail   : {agg([r for r in rs if r.window_kind == 'tail'])}")

    print("\n" + "-" * 78)
    print("계열별 집계")
    print("-" * 78)
    for fam, sts in families.items():
        rs = [r for r in records if r.sub_type in sts]
        print(f"\n{fam}")
        print(f"  forward: {agg([r for r in rs if r.window_kind == 'forward'])}")
        print(f"  tail   : {agg([r for r in rs if r.window_kind == 'tail'])}")


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("fall window weak-label transient dilution 진단 (읽기 전용)")
    print("=" * 78)
    print(f"source_dir (summary 기준): {SOURCE_DIR}")
    print(f"window: size={WINDOW_SIZE} stride={STRIDE}(→{WINDOW_SIZE}) tail_window={TAIL_WINDOW} "
          f"pad_short={PAD_SHORT}")
    print(f"rpca: max_iter={RPCA_MAX_ITER} tol={RPCA_TOL} | sdp: SUB_W={SUB_W} stride={SUB_STRIDE} "
          f"W_T={W_T} n_lags={N_LAGS} | zscore eps={ZSCORE_EPS}")
    print(f"fall range(세션 절대시간): {FALL_RANGE_S} s | rx={RX} n_sc={SAFESIGNAL_N_SUBCARRIERS}*2(both)")

    # ── 1) 파일 탐색 + (env,subject,sub-type) 표 ──
    by_subtype = discover_fall_files()
    print("\n[fall 파일 (env, subject, sub-type) 분포] side(_S) 제외")
    combos: dict[tuple[int, int], dict[str, int]] = {}
    for st, files in by_subtype.items():
        for p in files:
            m = parse_safesignal_filename(p)
            combos.setdefault((m.environment, m.subject), {}).setdefault(st, 0)
            combos[(m.environment, m.subject)][st] += 1
    header = "  env subj | " + " ".join(f"{st.replace('FALL_',''):>8}" for st in SUB_TYPES) + " | total"
    print(header)
    for key in sorted(combos):
        row = combos[key]
        cells = " ".join(f"{row.get(st, 0):>8}" for st in SUB_TYPES)
        tot = sum(row.values())
        print(f"   E{key[0]}  S{key[1]:02d} | {cells} | {tot}")
    has_s03 = any(k[1] == 3 for k in combos)
    print(f"\n  S03(subject=3) 존재: {has_s03} "
          f"({'우선 포함' if has_s03 else '미존재 → 가용 subject 대체'})")

    # ── 선택 ──
    selected = select_samples(by_subtype, rng)
    print(f"\n[선택 파일 — seed={SEED}, (env,subject) 조합당 {N_PER_SUBTYPE_PER_COMBO} trial]")
    total_sel = 0
    for st in SUB_TYPES:
        files = selected[st]
        total_sel += len(files)
        per_subj: dict[int, int] = {}
        for p in files:
            m = parse_safesignal_filename(p)
            per_subj[m.subject] = per_subj.get(m.subject, 0) + 1
        avail = len(by_subtype[st])
        print(f"  {st}: 선택 {len(files)}/{avail}  (subject별 {dict(sorted(per_subj.items()))})")
        for p in files:
            print(f"      {p.name}")
    print(f"  선택 총합: {total_sel} 파일")

    # ── 2.5) 등가검증 (표본 2파일) ──
    print("\n" + "=" * 78)
    print("등가검증 (training 경로 vs 서브함수 경로, cache row 순서 비의존)")
    print("=" * 78)
    eq_samples = [selected["FALL_SIT_F"][0], selected["FALL_WALK_F"][0]]
    eq_all_ok = True
    for sp in eq_samples:
        res = equivalence_check(sp)
        eq_all_ok = eq_all_ok and res["allclose"]
        print(f"  {res['filename']}: allclose={res['allclose']} max_abs_diff={res['max_abs_diff']:.3e} "
              f"shape={res['shape_train']} vs {res['shape_sub']}")
        print(f"      funcs: {res['funcs']}")
    if not eq_all_ok:
        print("\nFAIL: 등가검증 불일치 — pre-zscore 추출 경로가 학습 경로와 다름. 중단.")
        return 1
    print("\nOK: pre-zscore SDP 서브함수 경로 == 학습 경로 (z-score 적용 후 allclose).")

    # ── 모델 로드 (attention) ──
    print("\n" + "=" * 78)
    print("attention 분석용 checkpoint")
    print("=" * 78)
    model, ckpt_meta, ckpt_path = load_model_and_meta()
    if model is not None:
        print(f"  사용 checkpoint: {ckpt_meta['path']}")
        print(f"  classes={ckpt_meta['classes']} class_policy={ckpt_meta['class_policy']} "
              f"split={ckpt_meta['split']} threshold={ckpt_meta['threshold']}")
        print(f"  주의: within-subject(pretrained6) checkpoint. attention weights 는 T=28 "
              f"(subwindow) 축에 직접 매핑됨 → energy peak 와 동일 축에서 비교 가능.")
        print(f"        class head 는 weight 추출에 무관(forward hook 불필요, return_attention=True).")
    else:
        print("  checkpoint 미발견 → SDP 기반 분석만 진행, attention 컬럼은 비움.")

    # ── 2~4) window 처리 + 지표 ──
    print("\n" + "=" * 78)
    print("window 처리 (resample→RPCA→ACF→SDP, pre-zscore 지표 산출)")
    print("=" * 78)
    records: list[WindowRecord] = []
    anomalies: list[str] = []
    for st in SUB_TYPES:
        for path in selected[st]:
            meta = parse_safesignal_filename(path)
            raw = load_safesignal_csv(path, rx=RX)
            res = resample_to_100hz(raw.amplitude, raw.timestamps_us,
                                    target_hz=TARGET_HZ, max_gap_ms=MAX_GAP_MS)
            n = res.amplitude.shape[0]
            if n < WINDOW_SIZE:
                anomalies.append(f"  [skip<300] {path.name}: n={n} (orig_rate={res.original_rate_hz:.1f}Hz)")
                continue
            wins, starts, kinds = windows_with_starts(res.amplitude)
            if len(starts) != 2:
                anomalies.append(f"  [n_win!=2] {path.name}: n={n} starts={starts} kinds={kinds}")
            # zscore 입력 묶음 (attention 배치)
            sdps = [window_pre_zscore_sdp(w) for w in wins]
            zscored = np.stack([apply_global_zscore(s)[None, ...] for s in sdps], axis=0)  # (W,1,28,20)
            attn = attention_weights(model, zscored) if model is not None else None

            for wi, (w, s, kind, sdp) in enumerate(zip(wins, starts, kinds, sdps)):
                energy = subwindow_energy(sdp)
                peak_sw = int(np.argmax(energy))
                peak_t = subwindow_center_time_s(s, peak_sw)
                exp = expected_fall_subwindows(s)
                etot = float(energy.sum()) + 1e-12
                e_in_fall = float(energy[exp].sum()) / etot if exp else 0.0
                if attn is not None:
                    aw = attn[wi]
                    a_peak = int(np.argmax(aw))
                    a_t = subwindow_center_time_s(s, a_peak)
                    a_in = FALL_RANGE_S[0] <= a_t <= FALL_RANGE_S[1]
                else:
                    a_peak = a_t = a_in = None
                records.append(WindowRecord(
                    sub_type=st, filename=meta.filename, subject=meta.subject,
                    env=meta.environment, trial=meta.trial, window_kind=kind,
                    start_frame=s, n_frames=n,
                    sdp_mean_abs=float(np.abs(sdp).mean()), sdp_std=float(sdp.std()),
                    peak_subwindow=peak_sw, peak_time_s=peak_t,
                    peak_in_fall_range=(FALL_RANGE_S[0] <= peak_t <= FALL_RANGE_S[1]),
                    peak_energy_ratio=float(energy[peak_sw]) / etot,
                    energy_in_fall_ratio=e_in_fall,
                    expected_fall_subwindows=exp,
                    attn_peak=a_peak, attn_peak_time_s=a_t, attn_in_fall_range=a_in,
                    checkpoint_used=ckpt_path,
                    pre_zscore_sdp=sdp, energy=energy,
                ))
        print(f"  {st}: {sum(1 for r in records if r.sub_type == st)} windows 처리")

    if anomalies:
        print("\n[이상/주의 (계속 진행)]")
        for a in anomalies:
            print(a)

    # expected_fall_subwindows 예시 (forward/tail 대표)
    print("\n[expected_fall_subwindows 동적 계산 예시]")
    for kind in ("forward", "tail"):
        ex = next((r for r in records if r.window_kind == kind), None)
        if ex:
            print(f"  {kind} (start_frame={ex.start_frame}, n={ex.n_frames}): "
                  f"i={ex.expected_fall_subwindows} "
                  f"(center {subwindow_center_time_s(ex.start_frame, ex.expected_fall_subwindows[0]):.2f}"
                  f"~{subwindow_center_time_s(ex.start_frame, ex.expected_fall_subwindows[-1]):.2f}s)")

    # ── heatmap + CSV ──
    save_heatmaps(records)
    csv_path = OUT_DIR / "fall_window_dilution_summary.csv"
    cols = ["sub_type", "filename", "subject", "env", "trial", "window_kind", "start_frame",
            "n_frames", "sdp_mean_abs", "sdp_std", "peak_subwindow", "peak_time_s",
            "peak_in_fall_range", "peak_energy_ratio", "energy_in_fall_ratio",
            "attn_peak", "attn_peak_time_s", "attn_in_fall_range", "checkpoint_used"]
    lines = [",".join(cols)]
    for r in records:
        lines.append(",".join(str(v) for v in [
            r.sub_type, r.filename, r.subject, r.env, r.trial, r.window_kind, r.start_frame,
            r.n_frames, f"{r.sdp_mean_abs:.6f}", f"{r.sdp_std:.6f}", r.peak_subwindow,
            f"{r.peak_time_s:.3f}", int(r.peak_in_fall_range), f"{r.peak_energy_ratio:.4f}",
            f"{r.energy_in_fall_ratio:.4f}",
            "" if r.attn_peak is None else r.attn_peak,
            "" if r.attn_peak_time_s is None else f"{r.attn_peak_time_s:.3f}",
            "" if r.attn_in_fall_range is None else int(r.attn_in_fall_range),
            "" if r.checkpoint_used is None else Path(r.checkpoint_used).name,
        ]))
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    np.savez_compressed(
        OUT_DIR / "fall_window_energy.npz",
        energy=np.stack([r.energy for r in records]),
        sub_type=np.array([r.sub_type for r in records], dtype=object),
        window_kind=np.array([r.window_kind for r in records], dtype=object),
        filename=np.array([r.filename for r in records], dtype=object),
    )

    # ── 집계 ──
    summarize(records)

    print("\n[산출물]")
    print(f"  CSV    : {csv_path}")
    print(f"  npz    : {OUT_DIR / 'fall_window_energy.npz'}")
    print(f"  heatmap: {OUT_DIR}/sdp_heatmap_*.png  ({len(SUB_TYPES)} sub-type)")
    if ckpt_path:
        print(f"  checkpoint_used: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
