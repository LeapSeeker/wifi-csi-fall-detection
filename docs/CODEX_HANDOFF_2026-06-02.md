# SafeSignal Codex Handoff - 2026-06-02

This handoff is for continuing the SafeSignal fall-detection work from a fresh
Codex/Claude thread, especially on a school PC where the original Codex desktop
thread is unavailable.

## Start Prompt For New Thread

Paste this into the new web Codex/Claude thread after connecting the repository:

```text
AGENTS.md and docs/CODEX_HANDOFF_2026-06-02.md are the required starting context.
Read them first, then continue SafeSignal model work from Step2.

Do not assume prior chat history is available. Treat this handoff as the source
of conversation context, and treat the current repository code as the source of
truth for implementation.

Current priority:
1. Run/verify Step2 event-level operating point sweep for 6-class demo-primary.
2. Keep event sweep and forward/tail diagnostic paths separate:
   - event sweep: tail_window=False, realtime sliding simulation only.
   - forward/tail diagnostic: tail_window=True, training-cache 2-window simulation only.
3. Use held-out sessions only. Prefer stored split manifest if available; if not,
   regenerate split with train.py defaults and mark operating point as provisional.
4. Do not modify model files, training cache, or pipeline for Step2.
5. Save outputs under debug/modeling/diag_out/ and do not commit generated outputs.
```

## Repo And Environment Notes

- Repository: `LeapSeeker/wifi-csi-fall-detection`
- Local path used in this thread: `C:\Project\LastProject\wifi-csi-fall-detection`
- Branch at handoff time: `main`
- Remote at handoff time: `origin https://github.com/LeapSeeker/wifi-csi-fall-detection.git`
- Demo date: 2026-06-04
- Final presentation: 2026-06-11
- Demo primary policy: 6-class `pretrained6`, RUN excluded from demo operating metrics.

Fixed constraints:

- Model: CNN+GRU+Attention
- Signal chain: RPCA -> ACF -> SDP, no STFT
- Input: `(N, 1, 28, 20)`
- ACF lags: 1..20, lag0 excluded
- Window size: 300 frames at 100 Hz
- Pretrained/cache compatibility must be preserved for demo work.

## Current Local Worktree Caveat

At the time this handoff was created, the local worktree had uncommitted or
untracked experiment files, including:

- `model/finetune/train.py` modified by Codex to fix 6-class checkpoint metadata
  handling and argparse percent formatting.
- `debug/modeling/diag_fall_window_dilution.py`
- `debug/modeling/diag_event_sweep.py`
- `debug/modeling/diag_out/`
- `model/finetune/checkpoints_compare*_cpu*/`

This handoff commit intentionally records context only. Generated experiment
outputs should usually not be committed. If the `train.py` fix is needed on a
fresh machine, inspect the current repository state first; it may or may not
have been committed separately after this handoff.

## Key Decisions So Far

1. Use 6-class as the 2026-06-04 demo primary.
   - RUN/fast-walk is not planned for the demo.
   - 7-class remains reference-only.
   - If asked why RUN is absent: the design follows activity classes common in
     related fall-detection datasets/papers, while demo scope excludes fast
     walking/running.

2. Treat fall-window issue as weak-label transient dilution, not pure mislabel.
   - Fall sessions nominally use 1s wait + 2s fall + 1s post-fall still.
   - Actual cleaned/resampled fall sessions are about 5.5s, so tail windows are
     shifted toward post-fall stillness.

3. Do Step2 before Step3 training/cache changes.
   - Step2 is post-processing only and can reveal recall/FAR reachable without
     retraining.
   - Step3-a tail down-weight should be considered only after Step2 and
     forward/tail diagnostic.
   - Step3-b event-centered tail rebuilder is too risky for 6/4 unless Step2
     and Step3-a are insufficient.

## Codex 6-Class Vs 7-Class Check

Codex generated a 6-class cache from the 7-class SafeSignal cache:

- Source: `model/finetune/cache/safesignal_e1234_finetune7.npz`
- Derived: `model/finetune/cache/safesignal_e1234_pretrained6.npz`
- 6-class counts:
  - fall: 718
  - walking: 540
  - sit_stand: 524
  - lying: 539
  - standing: 360
  - picking: 360
- RUN removed: 540 windows

CPU-only 30-epoch comparison, same seed/settings, within-subject, augment,
threshold_min=0.10:

| Policy | Fall Recall | Fall F1 | FAR | Notes |
| --- | ---: | ---: | ---: | --- |
| 6-class pretrained6 | 0.736 | 0.660 | 0.152 | RUN removed |
| 7-class finetune7 | 0.646 | 0.583 | 0.143 | RUN included |

7-class false positive share included running at about 24.4%, supporting the
decision to use 6-class for demo-primary. These CPU runs are direction checks,
not final GPU benchmark numbers.

## Step1 Fall Window Dilution Diagnostic

Claude Code completed Step1 diagnostic. Codex reviewed the script and outputs.

Files reviewed:

- `debug/modeling/diag_out/fall_window_dilution_summary.csv`
- `debug/modeling/diag_out/fall_window_energy.npz`
- `debug/modeling/diag_out/sdp_heatmap_FALL_{SIT,STD,WALK}_{F,B}.png`
- `debug/modeling/diag_fall_window_dilution.py`

Important code facts:

- `model/preprocessing/window.py` uses `amplitude[-window_size:]` for
  `tail_window=True`.
- `debug/modeling/build_safesignal_cache.py` calls
  `preprocess_safesignal_file_full(..., stride=args.stride,
  tail_window=args.tail_window, ...)`.
- `collect/labels.py` documents the fall protocol as 1s pre-fall, 2s fall,
  1s post-fall still.

Diagnostic validation:

- Equivalence check compares `preprocess_safesignal_file_full` against the
  subfunction pre-zscore extraction path directly.
- It does not depend on cache row order.
- Reported allclose was true for the sampled files.

Output facts:

- CSV rows: 144 = 6 subtypes x 12 files x 2 windows.
- `n_frames`: min 550, max 555, mean about 552.
- tail `start_frame`: 250..255.
- Therefore tail is usually `[2.50..2.55s : 5.50..5.55s]`, not `[1.0s:4.0s]`.

Validated aggregate:

| Group | Window | energy_in_fall | attn_in_fall | peak_in_fall |
| --- | --- | ---: | ---: | ---: |
| SIT/STD | forward | 76.3% | 91.7% | 93.8% |
| SIT/STD | tail | 25.5% | 12.5% | 58.3% |
| WALK | forward | 73.3% | 91.7% | 87.5% |
| WALK | tail | 25.2% | 4.2% | 58.3% |

Interpretation:

- Primary cause is session geometry, not subtype difference.
- Tail is not pure junk: roughly half of peak positions still land near fall
  end/settling.
- But most energy/attention in tail is post-fall stillness.

## Step2 Final Prompt

Use this as the instruction for Claude Code/Codex if Step2 is not implemented
yet:

```text
[SafeSignal - event-level operating point sweep + forward/tail split diagnostic (no retraining)]

Background:
Use the existing within-subject model and tune only post-processing:
per-window threshold, N-consecutive confirmation, and margin. Find the
event/session-level operating point that maximizes fall recall while keeping FAR
under control. Recall is the primary metric. N is swept, not fixed, because N=2
can turn single-window fall spikes into FN.

Extra purpose:
Step1 showed fall sessions are about 5.5s after resampling, so the training
tail window is mostly post-fall stillness. Separately diagnose forward-only vs
tail-only recall/FN before deciding Step3-a tail down-weight.

Critical separation:
- Event sweep simulates realtime sliding inference: tail_window=False.
- Forward/tail diagnostic simulates training-cache 2-window fall construction:
  tail_window=True.
- Never include tail windows in event-level operating point metrics.

Execution stability:
- Run with python -u and tee logs, e.g.
  python -u debug/modeling/<script>.py 2>&1 | tee debug/modeling/diag_out/step2_run.log
- mkdir debug/modeling/diag_out with parents=True, exist_ok=True.
- Use matplotlib Agg backend immediately after import.

Primary setup:
- Demo primary is 6-class pretrained6.
- Use source_dir from
  model/finetune/cache/safesignal_e1234_finetune7.summary.json, usually
  data/cleaned.
- Use model/finetune/cache/safesignal_e1234_pretrained6.npz only for
  class_policy/classes/split reference. Do not use cache X as inference input.
- Actual inference input is held-out CSV rewindowed from source_dir.
- Prefer a GPU full-run 6-class within-subject best_operating.pt if available.
  Fallback:
  model/finetune/checkpoints_compare6_cpu/best_operating.pt
  The fallback is CPU 30-epoch validation only, not final performance.
- 7-class is reference-only and must not decide the primary operating point.

Class metadata:
- If cache/checkpoint classes or class_policy metadata exists, validate it.
- If both exist and conflict, stop.
- If metadata is missing, continue only after confirming model output dim == 6
  and fall index == 0. Emit warning.

Split/leakage:
- First search for stored train/val/test session manifest in checkpoint metadata,
  model/finetune/checkpoints*/ json/csv/txt, runs logs, wandb/tensorboard.
- If manifest exists, use it. Do not regenerate split.
- If not, call split_safesignal_within_subject with the same seed, val_ratio,
  and test_ratio, then print seed, ratios, session counts, and held-out sessions.
  Mark operating point provisional because it assumes the split matches 0b25a49.
- Use held-out sessions only. Exclude train/val from sweep metrics.
- Sanity check held-out window-level recall against the expected R around 0.674.
  Warn if it differs substantially.

Event sweep:
1. Find held-out session CSVs under source_dir.
2. Preprocess each session as training does:
   resample -> RPCA -> ACF -> SDP -> global z-score.
3. For event sweep only:
   - window=300
   - stride in {50, 100}
   - tail_window=False
   - pad_short=False
   - window_kind="sliding"
4. Compute time-ordered fall_prob and second_prob from softmax.
5. Positive rule:
   - margin off: fall_prob >= threshold_min
   - margin on: fall_prob >= threshold_min and fall_prob - second_prob >= m
6. Apply N-consecutive confirmation with N in {1,2}.
   If any confirmation fires in a session, the session is fall-detected.
7. Grid:
   threshold_min in {0.10,0.15,0.20,0.25,0.30}
   N in {1,2}
   margin in {off, on_m0.1, on_m0.2}
   stride in {50,100}
8. Metrics:
   event_recall = detected fall sessions / all fall sessions
   event_FAR = non-fall sessions with >=1 false fire / all non-fall sessions
   event_F1, TP, FP, FN, TN.
   Label event_FAR as RUN-excluded demo-range event-FAR.
9. Latency:
   confirmation_extra_latency_s = (N - 1) * stride / 100
   window_end_latency_s = (300 + (N - 1) * stride) / 100
   State that minimum window_end_latency is 3s due fixed 300-frame windows.
10. Select operating point:
   among FAR <= 0.15, maximize event_recall.
   tie-break: lower event_FAR, then N=1, then stride=50.
   Mark selected point provisional because held-out event count may be small.

Forward/tail split diagnostic:
- Use held-out sessions only. Exclude train/val.
- Separate from event sweep.
- Recreate training-cache 2-window construction:
  stride=None -> 300, tail_window=True, pad_short=False.
  forward = first [0:300]
  tail = amplitude[-300:]
- Use representative thresholds such as 0.20 and 0.30.
- Report:
  forward-only and tail-only window recall/FN for fall sessions.
  count of fall sessions where forward positive but tail negative.
  count of fall sessions where tail positive but forward negative.
  non-fall FP split by forward vs tail.
- Interpretation:
  If tail-only rescue sessions are rare and tail-only recall is low, Step3-a
  tail down-weight is likely safe. Otherwise be cautious.
- This diagnostic must not affect event operating point metrics.

Outputs:
- Script under debug/modeling/.
- Results under debug/modeling/diag_out/ and do not commit generated outputs.
- CSV: threshold_min, N, margin_mode, margin_value, stride, event_recall,
  event_FAR, event_F1, TP, FP, FN, TN, confirmation_extra_latency_s,
  window_end_latency_s.
- Forward/tail CSV: threshold, window_kind, window_recall, FN, FP,
  tail_only_rescue_sessions, forward_only_sessions.
- Session probability CSV preferred:
  session, activity, subject, env, trial, stride, window_start_frame,
  window_end_frame, window_kind, fall_prob, second_prob, margin, positive.
- Recall-FAR tradeoff plot.
- Console summary only. No STATE/markdown report.

Hard constraints:
- No retraining.
- Do not modify model files.
- Do not modify training cache.
- Use held-out sessions only.
- 300-frame input invariant.
- 6/4 primary is 6-class pretrained6.
- Selection criterion is max recall subject to event_FAR <= 0.15.
```

## Step3 Recommendation Pending Step2

After Step2:

1. If event-level sweep achieves demo recall/FAR target or acceptable rehearsal
   behavior, lock post-processing first.
2. If recall remains insufficient, try Step3-a:
   forward fall weight up, tail fall weight down.
   Suggested starting grid:
   - forward fall weight: 1.0
   - tail fall weight: 0.3 to 0.5
   - non-fall unchanged
3. Avoid Step3-b event-centered tail rebuilder before 6/4 unless necessary.
   It requires fall-specific window policy/cache rebuild and event timing
   assumptions. Stage metadata says nominal 1s/2s/1s but does not give measured
   event timestamp for each CSV.

## Useful Commands

Check repo status:

```powershell
$env:GIT_CONFIG_GLOBAL='NUL'
git -c safe.directory=C:/Project/LastProject/wifi-csi-fall-detection status --short
```

Run preprocessing tests:

```powershell
.\.codex-test-venv\Scripts\python.exe -m model.preprocessing.test_pipeline
```

Run server selfcheck:

```powershell
.\.codex-test-venv\Scripts\python.exe server\inference\_selfcheck.py
```

Run augmentation gate:

```powershell
.\.codex-test-venv\Scripts\python.exe verify_aug_gate.py
```

