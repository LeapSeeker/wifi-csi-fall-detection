"""SDP shift aug cache 실험 3개를 순차 실행."""
import subprocess, sys

PYTHON = sys.executable
SS_CACHE  = "model/finetune/cache/ss_peak160_p6.npz"
ALS_CACHE = "model/pretrained/checkpoints_backup_2026-05-18/dataset_cache_e12_w300_s300_tail_ps.npz"
AUG_CACHE = "model/finetune/cache/aug_fall_merged.npz"

COMMON = ["--class_policy", "pretrained6"]

experiments = [
    {
        "name": "f1f_sdp_shift_t15",
        "args": [
            "--safesignal_cache", SS_CACHE,
            "--alsaify_cache",    ALS_CACHE,
            "--aug_cache",        AUG_CACHE,
            "--max_aug_samples",  "500",
            "--fall_weight",      "2.0",
            "--ckpt_policy",      "f1_first",
            "--epochs",           "60",
            "--patience",         "15",
            "--ckpt_dir",         "model/finetune/checkpoints_f1f_sdp_shift_t15",
        ] + COMMON,
    },
    {
        "name": "rcf_sdp_shift_t15",
        "args": [
            "--safesignal_cache", SS_CACHE,
            "--alsaify_cache",    ALS_CACHE,
            "--aug_cache",        AUG_CACHE,
            "--max_aug_samples",  "500",
            "--fall_weight",      "2.0",
            "--ckpt_policy",      "recall_first",
            "--epochs",           "60",
            "--patience",         "15",
            "--ckpt_dir",         "model/finetune/checkpoints_rcf_sdp_shift_t15",
        ] + COMMON,
    },
    {
        "name": "rcf_sdp_shift_sr90",
        "args": [
            "--safesignal_cache", SS_CACHE,
            "--alsaify_cache",    ALS_CACHE,
            "--aug_cache",        AUG_CACHE,
            "--max_aug_samples",  "1000",
            "--fall_weight",      "2.0",
            "--source_ratio",     "0.9",
            "--ckpt_policy",      "recall_first",
            "--epochs",           "60",
            "--patience",         "15",
            "--ckpt_dir",         "model/finetune/checkpoints_rcf_sdp_shift_sr90",
        ] + COMMON,
    },
    {
        "name": "rcf_synthetic_t15",
        "args": [
            "--safesignal_cache", SS_CACHE,
            "--alsaify_cache",    ALS_CACHE,
            "--aug_cache",        "model/finetune/cache/aug_fall_synthetic.npz",
            "--max_aug_samples",  "500",
            "--fall_weight",      "2.0",
            "--ckpt_policy",      "recall_first",
            "--epochs",           "60",
            "--patience",         "15",
            "--ckpt_dir",         "model/finetune/checkpoints_rcf_synthetic_t15",
        ] + COMMON,
    },
]

for exp in experiments:
    print(f"\n=== Starting {exp['name']} ===", flush=True)
    cmd = [PYTHON, "-m", "model.finetune.train"] + exp["args"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        if "test within_subject" in line or "error" in line.lower():
            print(f"[{exp['name']}] {line}", flush=True)

print("\n=== ALL DONE ===", flush=True)
