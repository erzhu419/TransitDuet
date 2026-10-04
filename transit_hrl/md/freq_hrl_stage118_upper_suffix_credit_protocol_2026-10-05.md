# Stage117 Result And Stage118 Matched Upper Credit

Stage117 full run (`t135432`-`t135440`) passed the local-plan gate at both periods:

| Period | Cross-panel suffix gain | Bonferroni8 CI | Suffix versus option selection |
| --- | --- | --- | --- |
| 50 | +0.000174433 | [0.000118068, 0.000248208] | +0.000084385, CI [0.000043806, 0.000142642] |
| 100 | +0.000308124 | [0.000138099, 0.000514814] | +0.000067833, CI [0.000014428, 0.000130482] |

Gradient dot and cosine were also positive at both periods. This is small conditional
plan headroom under a frozen lower, not a trained-policy gain. It justifies testing
decision-to-episode-end credit rather than replacing the plan/lower interface first.

Stage118 trains two identical Stage116 complete eight-coordinate upper plan heads.
One uses option reward; the other uses the reverse cumulative option rewards aligned
to each upper decision. Pair initial weights, native scenarios, noise roles, eight updates,
Fisher radius, decoder and lower. Only the credit horizon differs. No critic or lower updates.
Check identical first-round rollouts and log cross-batch gradient agreement.

Evaluate final upper means on 32 fresh scenarios per root/period, with common lower noise.
Primary contrasts are suffix-minus-forecast and suffix-minus-option; blinded suffix must
execute exactly as forecast and is not a duplicate endpoint. Eight roots, two periods,
18,432 native episodes, 22,118,400 steps; bootstrap65,536, Bonferroni4. Both contrasts must
have positive CI lower bounds at both periods. Preflight: 96 episodes, 28,800 steps,
mechanical only. No best-iteration selection, seed extension or preflight tuning.

Schedule dynamically on node001-node006; weights and raw trajectories stay server-side.
If this fails, the supported local intervention has not transferred to score-gradient
learning; use gradient agreement and final paired returns to choose the next training change.

## Execution

Implementation revision: `34ab26669b`; 14 focused tests passed.
Native preflight `t135460` and independent aggregation `t135461` passed mechanically.
Frozen full run: `pointmaze_upper_suffix_credit_stage118_full_20261005_r1`, workers
`t135464`-`t135471`, aggregation `t135472`. All completed; all four corrected CIs
crossed zero and the formal gain gate was not supported. No preflight performance
metric changed the registered experiment. Stage119 addresses native gradient estimation.
