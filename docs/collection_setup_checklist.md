# SafeSignal Collection Setup Checklist

데이터 수집 직전 실험 조건을 고정하기 위한 체크리스트.

## Before Each Session Block

- TX antenna points toward the activity zone, not toward a wall.
- RX1/RX2 antenna directions are unchanged from the current environment setup.
- TX/RX positions, heights, and distances are not changed inside the same env-subject block.
- Server shows inference disabled during collection:
  `[Inference] disabled by SAFESIGNAL_DISABLE_INFERENCE=1`
- Pair quality is monitored after each saved CSV:
  - `pair_rate_hz`
  - `capture_ratio`
  - `pair_dt`
  - `ts_gap`
- If upload fails, keep the local CSV and retry upload later. Do not recollect only because Drive upload failed.

## Antenna Direction Mistake Policy

If TX/RX antenna direction was wrong for a collected block, do not delete the CSV files.

Recommended handling:

- Mark the affected activity/env/subject/trials as `condition_mismatch`.
- Exclude them from the primary fine-tuning/evaluation set until reviewed.
- Recollect the affected block when time allows, especially for fall classes.
- Keep the old files for robustness analysis or backup only.

Reason:

Directional antenna mismatch can change the CSI distribution even when packet quality looks normal.
The risk is highest when only one class was collected with the wrong orientation, because the model can
learn an antenna-condition artifact instead of the activity itself.

## Recollect Priority

1. Fall classes affected by antenna/position mistakes.
2. Fall classes with RECOLLECT or repeated WARN quality.
3. Walking / sit-stand if they are needed for false-positive control.
4. Other non-fall classes.
