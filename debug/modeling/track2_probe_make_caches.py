"""트랙2 PROVISIONAL probe — SafeSignal per-lag vs global 정규화 캐시 생성 (전처리 재실행 없음).

BANNER: PROVISIONAL mixed-normalization probe — SafeSignal per-lag vs global,
        Alsaify=global 고정, D-020 정식 ablation 아님.

raw SDP 캐시(normalization=none_raw_sdp)를 로드해 X 에 per-window self-normalization 두
방식을 각각 적용한 7-class 임시 npz 를 만들고, 기존 derive 스크립트로 6-class 파생한다.
- A안(global, control): pipeline.py:104 와 동일. 기존 finetune7 X 와 allclose 검증.
- B안(per-lag, mixed): axis=2(시간 28) per-lag, STD_FLOOR/EPS/CLIP 적용 + 분포 진단.
모든 정규화는 per-window self-normalization (train split 통계 fit 금지 → 누수 없음).

산출물은 전부 model/finetune/cache/ (gitignore) 아래 probe_ 접두. 커밋 금지.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BANNER = ("PROVISIONAL mixed-normalization probe — SafeSignal per-lag vs global, "
          "Alsaify=global 고정, D-020 정식 ablation 아님")

CACHE_DIR = PROJECT_ROOT / "model" / "finetune" / "cache"
RAW = CACHE_DIR / "safesignal_e1234_finetune7_rawsdp.npz"
FINETUNE7_REF = CACHE_DIR / "safesignal_e1234_finetune7.npz"   # A안 allclose 기준
DERIVE = PROJECT_ROOT / "debug" / "modeling" / "derive_pretrained6_cache.py"
DIAG = PROJECT_ROOT / "debug" / "modeling" / "diag_out"

# B안 상수 (명시 + 로그)
STD_FLOOR = 1e-4
EPS = 1e-6
CLIP = 3.0

# 보존할 키 (전부)
PRESERVE = ["X", "y", "subject", "source", "env", "filename", "activity",
            "trial", "within_file_index", "is_augmented", "classes", "class_policy"]

OUT_A7 = CACHE_DIR / "probe_safesignal_global7.npz"
OUT_B7 = CACHE_DIR / "probe_safesignal_perlag7.npz"
OUT_A6 = CACHE_DIR / "probe_safesignal_global6.npz"
OUT_B6 = CACHE_DIR / "probe_safesignal_perlag6.npz"


def banner(tag: str) -> None:
    print("=" * 100)
    print(f"[{tag}] {BANNER}")
    print("=" * 100)


def norm_global(X: np.ndarray) -> np.ndarray:
    """A안 global (pipeline.py:104 동일): per-window 전체 (28,20) mean/std."""
    m = X.mean(axis=(2, 3), keepdims=True)
    s = X.std(axis=(2, 3), keepdims=True)
    return ((X - m) / (s + EPS)).astype(np.float32)


def norm_perlag(X: np.ndarray):
    """B안 per-lag (mixed): axis=2(시간 28) per-lag self-norm + std_floor + clip.

    Returns (Xz, diag).
    """
    std_raw = X.std(axis=2, keepdims=True)            # (N,1,1,20) — lag별 시간축 std
    floored = std_raw <= STD_FLOOR
    std = np.maximum(std_raw, STD_FLOOR)
    mean = X.mean(axis=2, keepdims=True)
    z = (X - mean) / (std + EPS)
    clipped = np.abs(z) > CLIP
    z = np.clip(z, -CLIP, CLIP).astype(np.float32)
    diag = {
        "floored_ratio": float(floored.mean()),
        "clipped_ratio": float(clipped.mean()),
        "X_mean": float(z.mean()),
        "X_std": float(z.std()),
        "X_min": float(z.min()),
        "X_max": float(z.max()),
        "std_raw_min": float(std_raw.min()),
        "std_raw_med": float(np.median(std_raw)),
        "std_raw_max": float(std_raw.max()),
    }
    return z, diag


def save7(out: Path, src: np.lib.npyio.NpzFile, Xnew: np.ndarray, norm_value: str) -> None:
    payload = {}
    for k in PRESERVE:
        if k == "X":
            payload[k] = Xnew
        elif k in src.files:
            payload[k] = src[k]
        else:
            raise SystemExit(f"FAIL: raw 캐시에 키 누락: {k} (있는 키={src.files})")
    payload["normalization"] = norm_value  # scalar 메타
    np.savez_compressed(out, **payload)
    print(f"[write7] {out.name}  X={Xnew.shape} normalization={norm_value}")


def derive6(src7: Path, out6: Path) -> None:
    cmd = [sys.executable, str(DERIVE), "--src", str(src7), "--out", str(out6), "--overwrite"]
    print(f"[derive] {' '.join(cmd[1:])}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit(f"FAIL: derive 실패 ({out6.name})")


def reinject_norm6(out6: Path, norm_value: str, extra: dict) -> None:
    """derive 가 normalization scalar 를 보존하지 않으므로 6-class npz 에 재기입 + 사이드카 JSON."""
    d = np.load(out6, allow_pickle=True)
    payload = {k: d[k] for k in d.files}
    payload["normalization"] = norm_value
    np.savez_compressed(out6, **payload)
    side = out6.with_suffix(".probe_summary.json")
    side.write_text(json.dumps(
        {"banner": BANNER, "normalization": norm_value,
         "std_floor": STD_FLOOR, "eps": EPS, "clip": CLIP, **extra},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[reinject] {out6.name} normalization={norm_value}  (+{side.name})")


def verify6(out6: Path, norm_value: str) -> dict:
    d = np.load(out6, allow_pickle=True)
    X, y = d["X"], d["y"]
    classes = [str(c) for c in d["classes"].tolist()]
    counts = {classes[i]: int((y == i).sum()) for i in range(len(classes))}
    info = {
        "classes": classes, "shape": list(X.shape),
        "label_min": int(y.min()), "label_max": int(y.max()),
        "counts": counts, "normalization": str(d["normalization"]) if "normalization" in d.files else None,
        "X_std": float(X.std()),
    }
    ok = (classes == ["fall", "walking", "sit_stand", "lying", "standing", "picking"]
          and X.shape[1:] == (1, 28, 20) and y.min() == 0 and y.max() == 5)
    print(f"[verify6/{norm_value}] classes={classes} shape={tuple(X.shape)} "
          f"label_range=[{y.min()},{y.max()}] X.std={X.std():.5f} counts={counts} => {'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"FAIL: 6-class 검증 실패 ({out6.name})")
    return info


def main() -> int:
    banner("MAKE-CACHES")
    DIAG.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        raise SystemExit(f"FAIL: raw 캐시 없음: {RAW}")
    raw = np.load(RAW, allow_pickle=True)
    X_raw = raw["X"].astype(np.float32)
    print(f"[load] {RAW.name}  X_raw={X_raw.shape} normalization="
          f"{str(raw['normalization']) if 'normalization' in raw.files else '?'} "
          f"classes={[str(c) for c in raw['classes'].tolist()]} "
          f"is_augmented.any={bool(raw['is_augmented'].any())}")
    print(f"[consts] STD_FLOOR={STD_FLOOR} EPS={EPS} CLIP=[-{CLIP},{CLIP}]  "
          f"(per-window self-normalization; train-split-stat 미사용 → 누수 없음)")

    # ── A안 global ───────────────────────────────────────────────────────────
    Xa = norm_global(X_raw)
    save7(OUT_A7, raw, Xa, "global_self")

    # A안 정합 검증 (7-class, derive 전): 기존 finetune7 X 와 allclose
    ref = np.load(FINETUNE7_REF, allow_pickle=True)["X"].astype(np.float32)
    a_max = float(np.max(np.abs(Xa - ref)))
    a_ok = np.allclose(Xa, ref, atol=1e-4, rtol=1e-4)
    print(f"[A-allclose] global7 vs finetune7 X: max_abs_diff={a_max:.3e} allclose(1e-4)={a_ok}")
    if not a_ok:
        raise SystemExit("FAIL: A안 global 이 기존 finetune7 X 와 불일치 — 중단")

    # ── B안 per-lag ──────────────────────────────────────────────────────────
    Xb, bdiag = norm_perlag(X_raw)
    save7(OUT_B7, raw, Xb, "perlag_mixed_floor1e-4_clip3")
    print(f"[B-diag] floored_ratio={bdiag['floored_ratio']:.4%} "
          f"clipped_ratio={bdiag['clipped_ratio']:.4%} "
          f"X[mean/std/min/max]={bdiag['X_mean']:.4f}/{bdiag['X_std']:.4f}/"
          f"{bdiag['X_min']:.3f}/{bdiag['X_max']:.3f}  "
          f"std_raw[min/med/max]={bdiag['std_raw_min']:.4f}/{bdiag['std_raw_med']:.4f}/{bdiag['std_raw_max']:.4f}")

    # ── 6-class 파생 (기존 스크립트 호출, inline 재구현 금지) ────────────────
    derive6(OUT_A7, OUT_A6)
    reinject_norm6(OUT_A6, "global_self", {"arm": "A_global_control"})
    info_a = verify6(OUT_A6, "global_self")

    derive6(OUT_B7, OUT_B6)
    reinject_norm6(OUT_B6, "perlag_mixed_floor1e-4_clip3", {"arm": "B_perlag_mixed", **bdiag})
    info_b = verify6(OUT_B6, "perlag_mixed_floor1e-4_clip3")

    # ── cache summary 저장 (gitignore diag_out) ──────────────────────────────
    summary = {
        "banner": BANNER,
        "consts": {"STD_FLOOR": STD_FLOOR, "EPS": EPS, "CLIP": CLIP},
        "normalization_kind": "per-window self-normalization (no train-split-stat, no leakage)",
        "A_global": {"allclose_vs_finetune7": a_ok, "max_abs_diff": a_max, "verify6": info_a},
        "B_perlag": {"diag": bdiag, "verify6": info_b},
        "outputs": {"A7": OUT_A7.name, "B7": OUT_B7.name, "A6": OUT_A6.name, "B6": OUT_B6.name},
    }
    out_json = DIAG / "track2_probe_cache_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[summary] {out_json}")
    banner("MAKE-CACHES DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
