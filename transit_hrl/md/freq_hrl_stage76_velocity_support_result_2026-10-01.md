# Stage76: sampled upper velocity is mismatched to the cloned controller

All t118830..t118838 finished done0. Eight roots, 128 original BC label archives / 153,600 exact label states, 16 saved clone MSE reproductions and 512 Stage75 plan-energy frame matches. The audit evaluated 460,800 conditional actor rows and replayed 1,536,000 velocity rows. Native simulator steps, optimizer steps, forecaster fits and new trace/checkpoint writes were zero. Root computation took 2.38..2.89 seconds; only markers and about 49KB of compact JSON were pulled.

## Results

All four registered coverage contrasts are positive in every root and CI-supported under the frozen equal-root bootstrap / Bonferroni4 family. Fractions below are equal-root means; differences and CI are percentage points.

| Period / input test | Base % | Residual % | Difference [corrected CI] |
| --- | --- | --- | --- |
| 50 / above label speed q99 | 1.07 | 40.12 | +39.04 [36.03,41.33] |
| 50 / outside label coordinate ranges | 0.48 | 32.20 | +31.72 [30.40,33.10] |
| 100 / above label speed q99 | 1.15 | 12.16 | +11.00 [10.16,11.88] |
| 100 / outside label coordinate ranges | 0.47 | 8.26 | +7.80 [6.93,8.63] |

On historical BC states with the first 390 columns fixed, sampled-residual velocity raises mean command MSE from 0.003804 to 0.481523 at period50 and from 0.004678 to 0.130451 at period100. Mean root command-change RMS is 0.68795 / 0.34996; conditional Gaussian KL is 34.9533 / 4.0156. Zeroing velocity also worsens BC MSE to 0.234438 / 0.104025, so the controller depends on the base velocity signal. These actor-response quantities are descriptive, not additional CI claims.

Historical-only calibration ranges across roots: position-error q99 is 0.4966..0.6030 / 0.9452..1.0750; velocity-speed q99 is 1.2289..1.4112 / 1.2150..1.2541 for periods50/100. Full per-root calibrations and all four intervals are retained in `qualification_compact.json`; no action budget has been applied.

## Next

Preserve the causal base forecast and its velocity. Preregister one historical-data-only command-response budget for a coherent anchored residual curve, keeping reference and velocity mathematically coupled, then compare zero/original/calibrated execution on fresh native seeds. Avoid reward-selected scales and independent velocity clipping. Stage67 HOLD and the source policies remain unchanged.

## Limits

Marginal envelopes are not full joint-state support, and historical conditional response is not an on-policy reward experiment. Together with Stage75's native pathway intervention, this identifies a concrete upper exploration / lower training mismatch; it does not prove that a calibrated decoder improves reward or that joint Frequency-HRL training is repaired. Preflight r1's cleanup failure was corrected and archived; r2 passed the unchanged mechanical protocol.
