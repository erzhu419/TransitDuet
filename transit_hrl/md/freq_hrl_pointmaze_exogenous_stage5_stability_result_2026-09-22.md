# PointMaze Stage-5 Optimization-Stability Result

Date: 2026-09-22

## Outcome

All eight cells in
`pointmaze_exogenous_stage5_stability_v1_development_20260921_r1` completed.
The frozen development rule selected `more_rollouts`, authorizing a fresh-seed
Stage-5 V2 confirmation. This screen is post-hoc development selection, not
paper evidence.

| Arm | Worst-root success | Mean success | Mean return | Eligible |
|---|---:|---:|---:|---:|
| V1 control | 0.442 | 0.452 | 181.523 | no |
| dense-return rank | 0.442 | 0.452 | 181.523 | no |
| eight rollouts | 0.955 | 0.962 | 243.744 | yes |
| dense rank plus eight rollouts | 0.955 | 0.962 | 243.744 | yes |

Against the paired four-rollout control, the eight-rollout arm improved
tracking success by 0.527 and 0.494 on optimizer roots 134113 and 134127. Its
mean return gain was 62.221. Both roots also improved in success and return
relative to their paired untrained policies.

Changing checkpoint rank alone produced exactly the same selected iteration
and held-out rows as the V1 control. The two eight-rollout arms were likewise
identical. The selected repair is therefore the increased rollout batch, not
the alternative checkpoint rank. Ties resolve to the simpler V1-compatible
`success_then_return` rank.

## Execution Audit

The eight method/root signatures, fresh role seeds, 769-row histories,
16 final and 16 untrained episodes per cell, 134-dimensional states, 68,424
parameters, finite 300-step episodes, causal external-stream fields, and
runtime versions all matched the frozen protocol. Four-rollout cells made
18,432 gradient updates and eight-rollout cells made 24,576. Node003 through
node006 each ran two cells. Only compact result JSON files were synchronized.

## V2 Constraint

V2 must use fresh optimizer, train, selection, and evaluation seeds. Both flat
PPO and HRL must receive eight training rollout roots per iteration so their
environment-interaction budget remains equal. The network, task, reward,
horizon, rank mode, and registered Stage-5 gate remain unchanged.

## Claim Boundary

Allowed: increasing the rollout batch repaired both previously difficult
initializations under newly sampled environment paths and is the registered V2
recipe.

Forbidden: this selected-on-difficult-roots screen passes Stage 5, proves HRL
superior to flat PPO, authorizes frequency routing, or supplies independent
manuscript performance evidence.

