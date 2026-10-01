# Stage76: sampled upper velocity is mismatched to the cloned controller

All t118830..t118838 done0: eight roots, 153,600 BC states, 16 saved MSE reproductions and 512 exact native-plan frames. No native steps, optimization or new traces/checkpoints. Seven tests passed; only markers and about 49KB of compact JSON were pulled.

## Results

All four coverage contrasts are positive in every root and supported by the frozen Bonferroni4 root-bootstrap CI. Differences below are percentage points.

| Period / input test | Base % | Residual % | Difference [corrected CI] |
| --- | --- | --- | --- |
| 50 / above label speed q99 | 1.07 | 40.12 | +39.04 [36.03,41.33] |
| 50 / outside label coordinate ranges | 0.48 | 32.20 | +31.72 [30.40,33.10] |
| 100 / above label speed q99 | 1.15 | 12.16 | +11.00 [10.16,11.88] |
| 100 / outside label coordinate ranges | 0.47 | 8.26 | +7.80 [6.93,8.63] |

Holding the first 390 BC-state columns fixed, sampled velocity raises command MSE from 0.003804 to 0.481523 / 0.004678 to 0.130451 (period50/100). Command-change RMS is 0.68795 / 0.34996. Zeroing velocity also worsens MSE to 0.234438 / 0.104025. Training-only physical calibrations and complete response statistics are retained in `qualification_compact.json`.

## Next

Preserve base velocity; calibrate one coherent residual-curve command-response budget from historical labels, then compare zero/original/calibrated execution on fresh seeds. No reward-selected scale or independent velocity clipping. Stage67 HOLD remains unchanged.

## Limits

Marginal coverage and conditional response do not establish repaired native reward or full joint-state support. The calibrated decoder has not yet been applied.
