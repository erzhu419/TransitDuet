# Fixed Boundary Prediction Noise

Read-only attribution of the frozen parent-scaled predictors. Reuse all four
roots and resident windows/labels; no fitting, threshold selection, simulation,
controller loading or E8 access. One scheduler job reads caches on node001-node006.

For candidate p and baseline q, paired MSE difference is
mean(p^2-q^2)-2*mean((p-q)*mean_suffix(y)). The shared y^2 term cancels.
Given all A fits and realized windows, B suffix MC variance is
4/N^2 * sum((p-q)^2 * sample_var_B(y)/2). Report its SE, both individual B
replica effects and the largest window's variance share, not a normal CI.
Compare zero, constant and causal-t in training/whole-prefix-OOF views.

Report each prefix effect, fixed-prediction readout deletion, benefit shares,
and observed prefix dispersion minus estimated suffix noise. Retain negative
corrected dispersion. Preserve window halves and squared-prediction/alignment
terms. These are diagnostics, not new selection rules.

## Limits

Two B suffixes give noisy variance estimates. This SE excludes window redraw,
new-prefix, A-fitting and model-selection uncertainty. A enters other folds'
training, so A/B gap is not an independent-replica test. Prefix deletion does
not refit; fixed-prefix dispersion is not population generalization variance.
No suffix/window/prefix count is a new independent fitted policy. Previously
exposed development data cannot release confirmation or a policy claim.
No extra data or fits follow automatically; `PILOT_RECIPE_HOLD` remains.
