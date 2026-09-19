# PointMaze Goal-Control Stage-2 V2 Result

Date: 2026-09-19

## Evidence Status

The corrected development campaign
`pointmaze_goal_stage2_v2_development_20260919_r1` completed all 16 cells,
eight independent optimizer roots per method, and 256 held-out episodes. All
cells passed protocol, runtime, seed-role, capacity, finite-value, and option
boundary audits.

The ordinary-HRL learning gate is **not supported**. Multiscale enhancement
remains blocked.

## Registered Results

| Method | Success mean [95% CI] | Episode return mean [95% CI] | Final distance mean [95% CI] |
|---|---:|---:|---:|
| `flat_goal_ppo` | 0.297 [0.183, 0.411] | 79.432 [68.572, 90.292] | 0.946 [0.849, 1.043] |
| `hrl_goal_ppo` | 0.297 [0.152, 0.441] | 75.695 [62.512, 88.877] | 0.954 [0.816, 1.093] |

The registered HRL success lower bound was 0.152, below the required 0.50.
The paired HRL-minus-flat success difference was exactly 0.000 with 95% CI
[-0.137, 0.137]. Return and final-distance contrasts were also inconclusive.

## Repair Outcome

The V2 credit repair behaved as intended mechanically. Lower value losses at
registered checkpoint evaluations were generally around 0.002--0.011 instead
of the order-100 losses observed late in V1, and waypoint progress returns were
finite. That repair did not improve the registered task-success endpoint.

## Post-Hoc Reward Diagnosis

This diagnosis does not change the registered decision. Across the 128 held-out
episodes for each method, successful episodes terminated early and accumulated
far less positive environment dense reward than failed 300-step episodes:

| Method | Return, successful | Return, failed | Success-return correlation |
|---|---:|---:|---:|
| `flat_goal_ppo` | 47.810 | 92.784 | -0.500 |
| `hrl_goal_ppo` | 43.410 | 89.326 | -0.531 |

Thus the positive dense-reward sum used by both PPO trainers is not aligned
with the primary success endpoint under `continuing_task=False`: termination
removes future positive reward. A new protocol must give both methods the same
success-aligned control reward while retaining environment return as a reported
diagnostic. This reward correction must precede further hierarchy tuning.

## Claim Boundary

Allowed: V2 corrected lower option credit and reduced lower critic instability,
but ordinary HRL and the matched flat policy both remained weak on the primary
endpoint.

Forbidden: V2 supports ordinary-HRL learning, hierarchy superiority,
multiscale admission, or a positive Freq-HRL result.

Machine-readable analysis is under
`results/pointmaze_goal_stage2_v2_development_20260919_r1/analysis/`.
