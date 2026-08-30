# MuJoCo v14.27 orthogonal paired finite-difference outcome

## Frozen execution

- Source revision: `0e02a95853`
- Run: `mujoco_v14_27_orthogonal_paired_fd_preflight_20260830_r1`
- Scheduler tasks: `t84829`-`t84831`
- Placement: dynamic scheduler placement on `node004`; no required node and no Slurm
- Completion: all three tasks finished without retry or runtime error
- Evidence role: adaptive post-v14.26 preflight, not confirmatory evidence

Each synchronized cell passed its payload contract with 320 train-intervention
paths, 320 holdout-intervention paths, 64 design paths, and 64 untouched
validation paths for the selected candidate.

## Direction result

The orthogonal intervention solved the v14.26 estimator problem. Every upper
and lower design was full rank with condition number approximately one. All 24
mode-level train-versus-holdout direction cosines were positive.

| Environment | Upper overall cosine | Lower overall cosine | Upper minimum mode | Lower minimum mode |
|---|---:|---:|---:|---:|
| HalfCheetah-v5 | 0.9795 | 0.9943 | 0.9177 | 0.9137 |
| Hopper-v5 | 0.9993 | 1.0000 | 0.9628 | 0.9959 |
| Walker2d-v5 | 0.9662 | 0.9994 | 0.9301 | 0.9957 |

All six action-cost critics also retained positive holdout R2 and positive
action-permutation gain. The critic/direction gate therefore passed in all
three environments.

## Closed-loop validation

| Environment | Selected RMS | Design merit | Validation merit | Relative validation reduction | Reward violations | Result |
|---|---:|---:|---:|---:|---:|---|
| HalfCheetah-v5 | 1e-4 | 0.040729 | 0.899666 | -1523.90% | 3 | not supported |
| Hopper-v5 | 1e-4 | 0.054919 | 0.054714 | +1.24% | 0 | supported |
| Walker2d-v5 | 1e-4 | 0.051455 | 0.052839 | +4.63% | 0 | supported |

The common validation baseline merit was approximately `0.055402`. Hopper and
Walker improved independently with no reward-floor failure. HalfCheetah's only
design-eligible candidate did not transfer: its validation merit increased by
more than fifteen times baseline, its worst normalized frequency violation
rose to `0.5164`, and three reward floors failed.

## Decision

The all-environment preflight is **not supported** (`2/3` cells supported).
The orthogonal direction estimator itself is accepted as a solved development
component: its full-rank and cross-role agreement claims held in every cell.
The rejected component is a universal constant actor-bias restoration
transaction. HalfCheetah's long-horizon closed loop remains discontinuous
enough that pooled design selection overfit a candidate despite a stable native
cost direction.

The next protocol will not tune the actor direction or relax validation. It
will use one domain-general mechanism portfolio: function-preserving router
adaptation and orthogonal actor restoration are both evaluated by the same
closed-loop contract. Candidate eligibility must hold in two independent
design folds before pooled selection, followed by a fresh validation role.
This uses the complementary established boundaries: router adaptation was
stable in HalfCheetah/Hopper, while actor restoration is stable in
Hopper/Walker.
