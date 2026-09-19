# PointMaze Multiscale Stage-3 V2 Preflight

Date: 2026-09-19

## Outcome

The 20-cell preflight
`pointmaze_multiscale_stage3_v2_preflight_20260919_r1` completed as scheduler
tasks `t95931` through `t95950`. All tasks exited with code zero and wrote one
complete `result.json`. The run used algorithm revision
`67b9a12de0dd819183ebf201f7312aa97a23ea91`; only the compact JSON results were
synced locally.

The corrected state contract and execution protocol passed:

- all five methods and four scenarios were present exactly once;
- all episodes used the fixed 64-step preflight horizon;
- hierarchical cells made three upper decisions and three lower option
  boundaries, matching the 25-step macro period;
- both hierarchy levels reported the registered current-physical-feedback
  contract;
- trainable-parameter ratios ranged from 0.993071 to 1.006635;
- optimizer, training, selection, and evaluation seed roles were disjoint and
  matched across methods;
- all learned-policy updates and reported numeric values were finite;
- runtime versions matched the registered environment;
- clean cells reported zero stress in every channel;
- observation-noise cells reported only measurement noise;
- continuous-action-stress cells reported only slow and fast action stress;
- persistent-shift cells reported only persistent action stress;
- every exogenous stress diagnostic was exactly paired across methods within
  scenario, and disturbance truth remained unavailable to the actors.

One initial SSH launch attempt for `t95931` failed during key exchange; the
same unmodified task was rerouted and completed on node006. This did not alter
the preregistered cell or seed.

## Boundary

This is software and protocol evidence only. Two training updates, one
optimizer root, and one held-out episode per cell cannot estimate a performance
effect or a confidence interval. The preflight authorizes the frozen 160-cell
V2 development matrix without changing amplitudes, endpoints, seeds, model
settings, or claim gates.

