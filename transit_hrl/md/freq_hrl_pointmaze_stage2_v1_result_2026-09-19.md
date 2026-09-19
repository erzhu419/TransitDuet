# PointMaze Goal-Control Stage-2 V1 Result

Date: 2026-09-19

## Evidence Status

The registered development campaign
`pointmaze_goal_stage2_v1_development_20260919_r1` completed all 16 cells:
two methods, eight independent optimizer roots, and 128 held-out evaluation
episodes. All cells used the frozen runtime and passed protocol validation.

The ordinary-HRL learning gate is **not supported**. Multiscale enhancement
remains blocked.

## Registered Results

The optimizer root is the statistical unit. Intervals are two-sided 95% t
intervals over eight root-level means.

| Method | Success mean [95% CI] | Episode return mean [95% CI] | Final distance mean [95% CI] |
|---|---:|---:|---:|
| `flat_goal_ppo` | 0.234 [0.072, 0.397] | 76.038 [52.206, 99.870] | 1.143 [0.603, 1.682] |
| `hrl_goal_ppo` | 0.359 [0.197, 0.522] | 55.234 [45.379, 65.089] | 1.254 [1.044, 1.464] |

The frozen gate required the lower endpoint of the hierarchical success-rate
interval to be at least 0.50. The observed lower endpoint was 0.197.

The paired HRL-minus-flat success difference was +0.125 with 95% CI
[-0.181, 0.431]. The paired return difference was -20.804 with 95% CI
[-47.502, 5.895]. The paired final-distance improvement was -0.111 with 95%
CI [-0.739, 0.516]. All three paired contrasts are inconclusive.

## Post-Hoc Failure Diagnosis

This diagnosis does not change the registered decision. Across roots, lower
waypoint tracking generally deteriorated during training while lower value
loss grew. Inspection of the rollout path found that lower-level GAE crossed
fixed waypoint boundaries, so actions for one waypoint received returns from
later, different waypoints. The PointMaze rollout also used cumulative
negative absolute waypoint distance instead of the existing progress-based
intrinsic-reward contract. These are ordinary-HRL credit-assignment defects to
repair before another development gate; they are not evidence for multiscale
Freq-HRL.

## Claim Boundary

Allowed: V1 implemented a genuine waypoint/action hierarchy and produced a
valid negative development result. Its mean success exceeded the matched flat
mean, but neither the absolute learning gate nor the paired comparison was
supported.

Forbidden: V1 establishes a working ordinary-HRL substrate, a hierarchy
advantage, a multiscale advantage, or a positive Freq-HRL result.

Machine-readable analysis is under
`results/pointmaze_goal_stage2_v1_development_20260919_r1/analysis/`.
