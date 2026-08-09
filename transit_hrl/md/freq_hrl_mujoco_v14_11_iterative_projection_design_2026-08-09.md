# MuJoCo v14.11 Iterative Deployment Projection

Date: 2026-08-09

## Decision

The v14.10 source-bound HalfCheetah preflight validated deployment-aligned
frequency accounting and exact causal-router calibration, but all learned arms
selected the initial checkpoint. Accepted one-step corrections reduced
same-batch frequency power by only about 0.25% to 0.87%, far below the
registered 5% or 10% paired targets. Additional dual-rate tuning is therefore
not authorized on the v14.10 update rule.

v14.11 replaces the single post-PPO correction with an iterative projection.
Every step is checked against the same pre-projection PPO reward-loss baseline,
so a tolerance of `1e-4` is a cumulative budget, not a per-step budget that can
be spent repeatedly. Iteration stops when the registered frequency target is
reached, the cumulative reward guard rejects a candidate, or the frozen step
limit is exhausted.

## Frozen Identity

- Shared-core protocol:
  `freq_hrl_mujoco_shared_core_v14_11_iterative_deployment_projection`
- Algorithm revision:
  `ed87b9a0d3e2b78a9c7c10fd76291c64af246564`
- Freq-HRL source manifest SHA-256:
  `97f2cbfe15bc5d5ae061a2c23a9159202ba9f20eec85d94e2bb2cf52ee475863`
- Development protocol:
  `mujoco_v14_11_iterative_projection_screen_v1`

The launcher and experiment spec have their own runtime revision and file
hashes in `preregistration.json`. The algorithm revision above identifies the
committed `transit_hrl/freq_hrl` bytes and is verified again inside every cell.

## Preflight Arms

All continuations load the same environment/optimizer-specific anchor,
preserve the v14.10 paired-relative selector, use router strength `0.50`, and
use fresh training, selection, evaluation, and optimizer seed roles.

| Role | Projection configuration |
|---|---|
| mean control | router strength 0.00, no deployment dual |
| router calibration | router strength 0.50, no deployment dual, mean-reward selector |
| matched comparator | router strength 0.50, no deployment dual, paired-relative selector |
| legacy learned | upper/lower dual 0.03/0.08, step scale 1/3, target 5%, `k=1` |
| iterative learned | same settings with `k=4`, `k=8`, or `k=16` |
| aggressive iterative | step scale 3/10, target 5%, `k=8` |
| aggressive 10% target | step scale 3/10, target 10%, `k=8` |

The preflight scope is one fresh optimizer seed on HalfCheetah-v5: one anchor
plus nine continuation cells. The full 480-cell development screen is not
authorized before preflight analysis.

## Mechanism Gates

An iterative arm is eligible for expansion only when all existing v14.10
provenance, calibration, trained-checkpoint, actor/action-change, reward-floor,
and five-frequency-endpoint gates pass, and all of the following hold:

1. At least two projection steps are accepted in one update.
2. At least one update is genuinely multi-step.
3. No final reward-loss delta exceeds the one cumulative registered budget.
4. Requested step counts and reward tolerances in training history match the
   frozen arm spec.
5. Mean within-update frequency-power reduction exceeds the matched `k=1`
   arm by at least one percentage point.
6. A trained checkpoint, not iteration `-1`, is selected.

A constraint already inside its target is recorded as feasible and is not
forced to update. A violating constraint without an accepted reducing step is
recorded as a mechanism failure. Exhausting the step budget without reaching
the target remains explicit in the training diagnostics.

## Local Verification

The shared-core unit comparison held model state and trajectory fixed. Four
steps reduced lower-band power from `0.21240076` to `0.11926103`, compared with
`0.18757328` after one step, without exceeding the cumulative reward guard.

A real two-stage HalfCheetah-v5 smoke loaded the paired anchor and requested
eight upper/lower projection steps. The lower constraint accepted two steps,
reduced power from `5.0687e-6` to `1.8783e-6`, crossed its paired target
`2.0242e-6`, and had cumulative reward-loss delta `7.8753e-6` under the
registered `1e-4` budget. The upper constraint was already feasible and made
zero updates. This is implementation evidence only.

## Claim Boundary

The implementation, unit comparison, and local smoke do not support reward
improvement, learned frequency separation, no-tradeoff behavior, cross-task
generality, or publication claims. The single-seed scheduler preflight may
reject the mechanism or authorize a larger development screen; it cannot
provide confidence intervals or confirmatory evidence.
