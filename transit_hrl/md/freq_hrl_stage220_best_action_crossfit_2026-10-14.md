# Stage220: learned decisions without an oracle inference menu

## Protocol

Original240 states, four matched views; complete-prefix60/60 train/held splits. Both future partitions and both directions use four training draws only. No disputed-state filtering, held-input normalization or future menu at inference.

Compare exact98-feature paired-ridge against safe-best-action CE; shared GLM plus linear/nonlinear/conditioned wait residuals. New branches use600 matched AdamW steps. Report same/other futures on train/held prefixes and same-state single/mixed comparisons.

## Results

- `t151863-t151870` fits and `t151884` report: all exit0,32 cells qualified,6 tests passed.19,200 decision and57,600 wait-residual updates; no new native episodes or RL updates. Pulled248,223 report bytes, no raw data/models.
- Own-menu best actions disagree in152/240 and142/240 states, versus66/60 under the common-safe menu in Stage219. Menu flips add168/294 disagreements (57.1%); hard labels amplify safety-boundary noise.
- Equal-view mean physical wait-mass log-loss, GLM/conditioned: train same0.030381/0.016648; train other0.035117/0.047713; held other0.041557/0.060782. All residual heads have worse held-other risk than GLM in all four views. Conditioning is active (minimum context-weight norm4.707), not an unused path.
- CE has worse held-other cost than paired-ridge with the same linear gate in3/4 views. With paired-ridge, conditioned gating gives negative cost/wait deltas in8/8 mixed-view future-direction aggregates (linear6/8, unconditioned nonlinear3/8). This is useful local control evidence, not uniform calibration or cost dominance.

## Next

Retain the proper cost critic. Replace hard winner imitation with physical cost-gap/regret-sensitive learning; soften the noisy safety boundary and calibrate/shrink wait corrections using training-only future splits. Keep every held state.

## Boundary

Development counterfactuals under the original greedy suffix; reused future sensitivities are not independent seed CI or end-to-end HRL validation. Exact training fit is not the current bottleneck.
