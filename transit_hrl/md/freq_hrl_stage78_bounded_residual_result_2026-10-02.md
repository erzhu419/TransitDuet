# Stage78: constraint met; P100 performance gap remains

Preflight t119795/t119796 and full t119852-t119860 all exited 0; 12 focused
tests passed. Full: eight roots, 2,048 episodes / 2,457,600 native steps.
All 24 pairwise reward/tracking endpoints used Bonferroni24 root-bootstrap CIs.

| Period | Reward contrast | Mean | Corrected CI |
|---|---|---:|---|
| 50 | Bounded - Stage77 ratio | +1.7404 | [1.0565, 2.4416] |
| 50 | Bounded - zero | -0.0811 | [-0.3015, 0.1375] |
| 100 | Bounded - Stage77 ratio | +2.1195 | [1.4125, 3.0878] |
| 100 | Bounded - zero | -0.8920 | [-1.5940, -0.3202] |

All 16 historical constraints passed: command-change RMS / BC RMSE was
0.570-0.672 (P50), 0.627-0.777 (P100). Each period used three/two response
evaluations respectively; 40 total, 691,200 actor rows / 1,152 batches, fully
counted. Tracking versus zero was inconclusive at P50, worse at P100.

## Limitations / Next Step

A historical aggregate mean-command constraint is not a native reward guarantee.
P50 uncertainty is not noninferiority or improvement; P100 retains supported
harm. Actors/critics/Adam/forecaster stayed frozen. Stage67 HOLD remains.
Freeze this feasible decoder for the next native credit/directional-response
diagnosis at both periods; do not shrink alpha further on evaluation rewards
or adopt a learned-HRL/frequency claim. Compact evidence is in
`results/pointmaze_bounded_residual_stage78_full_20261002_r1/qualification_compact.json`.
