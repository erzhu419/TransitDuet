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

## Result And Next Step

Task t214610, code `80133d5bca`: all four frozen losses reproduced, zero new
fits/native steps. Eight focused tests passed; only 70,021 public bytes returned.
See [moments](../results/plan_validity_boundary_noise_20261111/summary.json) and
[receipt](../results/plan_validity_boundary_noise_20261111/release.json).

Whole-prefix-OOF paired MSE difference versus zero, negative improves:

| Task/root | B difference | Suffix MC SE | B-improving prefixes |
| --- | ---: | ---: | ---: |
| PointMaze/1972110000 | -0.0003273 | 0.0000722 | 7/8 |
| PointMaze/1972120000 | -0.0000459 | 0.0000283 | 5/8 |
| Reacher/1972210000 | -0.0001546 | 0.0007719 | 4/8 |
| Reacher/1972220000 | +0.0008828 | 0.0011958 | 1/8 |

PointMaze root1 improves after every single-prefix deletion and on both B
replicas; its largest prefix supplies 28.6% of net benefit. Root2's B replicas
disagree and one prefix supplies 93.7% of its small net benefit.

Reacher root1's effect is 0.20 estimated suffix SE; one window supplies 81.8%
of variance, deletion can reverse the mean and corrected prefix dispersion is
negative. Yet prefix 1972275005 is harmful on both A/B (+0.004393/+0.003573).
Root2 has 7/8 harmful prefixes, two harmful training-B replicas and 89.5% of
observed fixed-prefix dispersion remaining after estimated suffix correction.
Thus additional B precision alone cannot address the observed problems.

Next hold geometry/DF fixed: compare more independent A futures with wider
training-prefix coverage at a matched declared budget, then use a separate
new-prefix/new-future readout. No such sampling or fitting is released here.
