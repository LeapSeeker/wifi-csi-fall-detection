"""트랙1 D-020 FORMAL ablation — SafeSignal+Alsaify 둘 다 동일 정규화 캐시 생성.

BANNER: D-020 FORMAL ablation — SafeSignal+Alsaify 둘 다 동일 정규화,
        A=both global / B=both per-lag.

두 raw 캐시(normalization=none_raw_sdp)에서 X 에 per-window self-normalization 적용
(train-split-stat 미사용 → 누수 없음). A=global, B=per-lag(std_floor+clip).
- SafeSignal: raw 7-class(finetune7) → 정규화 → derive_pretrained6_cache.py 로 6-class.
- Alsaify: 이미 6-class(0..5) → derive 금지, raw 6-class 에 정규화만 적용.
A/B 는 동일 raw·동일 행순서를 공유하고 정규화만 다름 → 깨끗한 paired.

산출물 전부 gitignore 경로(model/finetune/cache/, model/pretrained/checkpoints/) track1_ 접두.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BANNER = ("D-020 FORMAL ablation — SafeSignal+Alsaify 둘 다 동일 정규화, "
          "A=both global / B=both per-lag")

CACHE_DIR = PROJECT_ROOT / "model" / "finetune" / "cache"
CKPT_DIR = PROJECT_ROOT / "model" / "pretrained" / "checkpoints"
DIAG = PROJECT_ROOT / "debug" / "modeling" / "diag_out"
DERIVE = PROJECT_ROOT / "debug" / "modeling" / "derive_pretrained6_cache.py"

SS_RAW = CACHE_DIR / "safesignal_e1234_finetune7_rawsdp.npz"
SS_REF = CACHE_DIR / "safesignal_e1234_finetune7.npz"          # A안 allclose 기준
ALS_RAW = CKPT_DIR / "dataset_cache_e12_w300_s300_lag1_20_tail_ps_rawsdp.npz"

STD_FLOOR = 1e-4
EPS = 1e-6
CLIP = 3.0

SS_PRESERVE = ["X", "y", "subject", "source", "env", "filename", "activity",
               "trial", "within_file_index", "is_augmented", "classes", "class_policy"]

# SafeSignal 출력
SS_A7 = CACHE_DIR / "track1_ss_global7.npz"
SS_B7 = CACHE_DIR / "track1_ss_perlag7.npz"
SS_A6 = CACHE_DIR / "track1_ss_global6.npz"
SS_B6 = CACHE_DIR / "track1_ss_perlag6.npz"
# Alsaify 출력 (derive 없음)
ALS_A6 = CKPT_DIR / "track1_als_global6.npz"
ALS_B6 = CKPT_DIR / "track1_als_perlag6.npz"

EXPECT6 = ["fall", "walking", "sit_stand", "lying", "standing", "picking"]


def banner(tag: str) -> None:
    print("=" * 104)
    print(f"[{tag}] {BANNER}")
    print("=" * 104)


def norm_global(X):
    m = X.mean(axis=(2, 3), keepdims=True)
    s = X.std(axis=(2, 3), keepdims=True)
    return ((X - m) / (s + EPS)).astype(np.float32)


def norm_perlag(X):
    std_raw = X.std(axis=2, keepdims=True)
    floored = std_raw <= STD_FLOOR
    std = np.maximum(std_raw, STD_FLOOR)
    z = (X - X.mean(axis=2, keepdims=True)) / (std + EPS)
    clipped = np.abs(z) > CLIP
    z = np.clip(z, -CLIP, CLIP).astype(np.float32)
    diag = {"floored_ratio": float(floored.mean()), "clipped_ratio": float(clipped.mean()),
            "X_mean": float(z.mean()), "X_std": float(z.std()),
            "X_min": float(z.min()), "X_max": float(z.max()),
            "std_raw_min": float(std_raw.min()), "std_raw_med": float(np.median(std_raw)),
            "std_raw_max": float(std_raw.max())}
    return z, diag


def self_consistent_global(X_raw) -> dict:
    """raw 에 global 적용 시 윈도우별 mean≈0, std≈1 검증 (Alsaify A안 정합)."""
    Xg = norm_global(X_raw)
    n = Xg.shape[0]
    means = np.array([float(Xg[i, 0].mean()) for i in range(n)])
    stds = np.array([float(Xg[i, 0].std()) for i in range(n)])
    return {"z_mean_absmax": float(np.abs(means).max()),
            "z_std_min": float(stds.min()), "z_std_med": float(np.median(stds)),
            "z_std_max": float(stds.max())}


def save_with(out: Path, src, Xnew, preserve, norm_value, extra_class_policy=None):
    payload = {}
    for k in preserve:
        if k == "X":
            payload[k] = Xnew
        elif k in src.files:
            payload[k] = src[k]
        else:
            raise SystemExit(f"FAIL: {out.name} 키 누락: {k} (있는키={src.files})")
    payload["normalization"] = norm_value
    np.savez_compressed(out, **payload)
    print(f"[write] {out.name}  X={Xnew.shape} normalization={norm_value}")


def derive6(src7: Path, out6: Path):
    cmd = [sys.executable, str(DERIVE), "--src", str(src7), "--out", str(out6), "--overwrite"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[derive] {out6.name}: {res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ''}")
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit(f"FAIL: derive 실패 {out6.name}")


def reinject(out6: Path, norm_value: str, extra: dict):
    d = np.load(out6, allow_pickle=True)
    payload = {k: d[k] for k in d.files}
    payload["normalization"] = norm_value
    np.savez_compressed(out6, **payload)
    side = out6.with_suffix(".track1_summary.json")
    side.write_text(json.dumps({"banner": BANNER, "normalization": norm_value,
                                "std_floor": STD_FLOOR, "eps": EPS, "clip": CLIP, **extra},
                               indent=2, ensure_ascii=False), encoding="utf-8")


def verify6(out6: Path, exp_windows: int, tag: str) -> dict:
    d = np.load(out6, allow_pickle=True)
    X, y = d["X"], d["y"]
    classes = [str(c) for c in d["classes"].tolist()]
    counts = {classes[i]: int((y == i).sum()) for i in range(len(classes))}
    ok = (classes == EXPECT6 and X.shape[1:] == (1, 28, 20)
          and int(y.min()) == 0 and int(y.max()) == 5 and X.shape[0] == exp_windows)
    print(f"[verify6/{tag}] classes={classes} shape={tuple(X.shape)} "
          f"label=[{int(y.min())},{int(y.max())}] X.std={float(X.std()):.5f} counts={counts} "
          f"=> {'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"FAIL: 6-class 검증 실패 {tag}")
    return {"classes": classes, "shape": list(X.shape), "counts": counts,
            "label_range": [int(y.min()), int(y.max())], "X_std": float(X.std())}


def main() -> int:
    banner("TRACK1 MAKE-CACHES")
    DIAG.mkdir(parents=True, exist_ok=True)
    print(f"[consts] STD_FLOOR={STD_FLOOR} EPS={EPS} CLIP=[-{CLIP},{CLIP}]  "
          f"(per-window self-normalization; train-split-stat 미사용 → 누수 없음)")

    ss = np.load(SS_RAW, allow_pickle=True)
    als = np.load(ALS_RAW, allow_pickle=True)
    Xss = ss["X"].astype(np.float32)
    Xals = als["X"].astype(np.float32)
    print(f"[load] SS raw {Xss.shape} norm={str(ss['normalization'])}  "
          f"Alsaify raw {Xals.shape} norm={str(als['normalization'])}")
    als_preserve = list(als.files)  # X,y,subject,env,filename,classes,(normalization,lag_policy)

    summary = {"banner": BANNER, "consts": {"STD_FLOOR": STD_FLOOR, "EPS": EPS, "CLIP": CLIP},
               "normalization_kind": "per-window self-normalization (no train-split-stat)"}

    # ── A안 global ───────────────────────────────────────────────────────────
    print("\n--- A안 (both global) ---")
    Xss_a = norm_global(Xss)
    save_with(SS_A7, ss, Xss_a, SS_PRESERVE, "global_self")
    # (i) SafeSignal A안 allclose vs finetune7
    ref = np.load(SS_REF, allow_pickle=True)["X"].astype(np.float32)
    a_max = float(np.max(np.abs(Xss_a - ref)))
    a_ok = np.allclose(Xss_a, ref, atol=1e-4, rtol=1e-4)
    print(f"[verify-i] SS global vs finetune7: max_abs_diff={a_max:.3e} allclose(1e-4)={a_ok}")
    if not a_ok:
        raise SystemExit("FAIL: SafeSignal A안 global 이 finetune7 과 불일치 — 중단")
    Xals_a = norm_global(Xals)
    save_with(ALS_A6, als, Xals_a, als_preserve, "global_self")
    # (iii) Alsaify A안 self-consistent
    sc = self_consistent_global(Xals)
    sc_ok = sc["z_mean_absmax"] < 1e-4 and abs(sc["z_std_med"] - 1.0) < 1e-2
    print(f"[verify-iii] Alsaify global self-consistent: |mean|max={sc['z_mean_absmax']:.2e} "
          f"std[min/med/max]={sc['z_std_min']:.5f}/{sc['z_std_med']:.5f}/{sc['z_std_max']:.5f} "
          f"=> {'OK' if sc_ok else 'FAIL'}")
    if not sc_ok:
        raise SystemExit("FAIL: Alsaify A안 self-consistent 실패")

    # ── B안 per-lag ──────────────────────────────────────────────────────────
    print("\n--- B안 (both per-lag) ---")
    Xss_b, ss_bdiag = norm_perlag(Xss)
    save_with(SS_B7, ss, Xss_b, SS_PRESERVE, "perlag_floor1e-4_clip3")
    print(f"[verify-ii/SS]  floored={ss_bdiag['floored_ratio']:.4%} clipped={ss_bdiag['clipped_ratio']:.4%} "
          f"X[mean/std/min/max]={ss_bdiag['X_mean']:.4f}/{ss_bdiag['X_std']:.4f}/"
          f"{ss_bdiag['X_min']:.3f}/{ss_bdiag['X_max']:.3f} "
          f"std_raw_med={ss_bdiag['std_raw_med']:.4f}")
    Xals_b, als_bdiag = norm_perlag(Xals)
    save_with(ALS_B6, als, Xals_b, als_preserve, "perlag_floor1e-4_clip3")
    print(f"[verify-ii/ALS] floored={als_bdiag['floored_ratio']:.4%} clipped={als_bdiag['clipped_ratio']:.4%} "
          f"X[mean/std/min/max]={als_bdiag['X_mean']:.4f}/{als_bdiag['X_std']:.4f}/"
          f"{als_bdiag['X_min']:.3f}/{als_bdiag['X_max']:.3f} "
          f"std_raw_med={als_bdiag['std_raw_med']:.4f}")

    # ── SafeSignal derive 6-class (Alsaify 는 derive 금지) ───────────────────
    print("\n--- SafeSignal derive → 6-class ---")
    derive6(SS_A7, SS_A6); reinject(SS_A6, "global_self", {"arm": "A_both_global"})
    derive6(SS_B7, SS_B6); reinject(SS_B6, "perlag_floor1e-4_clip3", {"arm": "B_both_perlag", **ss_bdiag})

    # ── (iv) 각 산출 캐시 검증 ───────────────────────────────────────────────
    print("\n--- (iv) 산출 캐시 검증 ---")
    v = {
        "SS_A6": verify6(SS_A6, 3041, "SS_global6"),
        "SS_B6": verify6(SS_B6, 3041, "SS_perlag6"),
        "ALS_A6": verify6(ALS_A6, 8326, "ALS_global6"),
        "ALS_B6": verify6(ALS_B6, 8326, "ALS_perlag6"),
    }

    summary.update({
        "verify_i_SS_global_allclose": {"ok": a_ok, "max_abs_diff": a_max},
        "verify_ii_SS_perlag_diag": ss_bdiag,
        "verify_ii_ALS_perlag_diag": als_bdiag,
        "verify_iii_ALS_global_selfconsistent": {"ok": sc_ok, **sc},
        "verify_iv": v,
        "outputs": {"SS_A6": SS_A6.name, "SS_B6": SS_B6.name,
                    "ALS_A6": ALS_A6.name, "ALS_B6": ALS_B6.name},
    })
    out_json = DIAG / "track1_formal_cache_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[summary] {out_json}")
    banner("TRACK1 MAKE-CACHES DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
