# PointMaze Goal-Control Stage-2 V3 Result

Date: 2026-09-19

## Evidence Status

The reward-aligned development campaign
`pointmaze_goal_stage2_v3_development_20260919_r1` completed all 16 cells,
eight independent optimizer roots per method, and 256 held-out fixed-horizon
episodes. All cells passed protocol, runtime, seed-role, capacity, finite-value,
and option-boundary audits.

The registered ordinary-HRL learning gate is **supported**. A separately
registered multiscale factorial experiment is now admitted. This result alone
is not evidence for a multiscale or Freq-HRL advantage.

## Registered Results

| Method | Success mean [95% CI] | Dense return mean [95% CI] | Final distance mean [95% CI] |
|---|---:|---:|---:|
| `flat_goal_ppo` | 0.719 [0.579, 0.858] | 144.874 [127.245, 162.503] | 0.752 [0.599, 0.905] |
| `hrl_goal_ppo` | 0.789 [0.710, 0.868] | 178.147 [158.395, 197.899] | 0.526 [0.393, 0.659] |

The registered HRL success lower bound was 0.710, above the required 0.50.

Paired HRL-minus-flat results over optimizer-root means were:

- success: +0.070, 95% CI [-0.043, 0.184], inconclusive;
- dense return: +33.273, 95% CI [10.523, 56.023], supported;
- final-distance improvement: +0.226, 95% CI [0.038, 0.414], supported.

Thus V3 establishes a functioning ordinary goal-conditioned hierarchy but does
not establish a success-rate advantage over flat PPO.

## Reward and Credit Diagnostics

All held-out episodes ran for 300 steps with `continuing_task=True` and an
unchanged target. Successful episodes now had substantially larger dense return
than failed episodes. Episode-level success-return correlations were +0.742 for
flat PPO and +0.933 for HRL, reversing the V2 objective conflict.

The V2 lower-credit repair remained stable. Across registered checkpoint
evaluations, the largest root-level lower value loss was below 0.014, waypoint
progress returns were finite, and every held-out HRL episode recorded exactly
12 lower option boundaries.

## Claim Boundary

Allowed: under the V3 fixed-horizon PointMaze protocol, ordinary
goal-conditioned HRL learned the task with a root-level success CI entirely
above the registered threshold. Its return and final distance improved over the
capacity-matched flat policy, while its success-rate advantage was
inconclusive.

Forbidden: V3 proves a multiscale representation benefit, a frequency-by-
hierarchy interaction, general Freq-HRL superiority, AntMaze performance, or a
confirmatory result.

Machine-readable analysis is under
`results/pointmaze_goal_stage2_v3_development_20260919_r1/analysis/`.
