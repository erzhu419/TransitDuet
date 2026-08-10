# MuJoCo v14.15 Closed-Loop Restoration-Filter Development Protocol

## Motivation

The frozen v14.14 HalfCheetah preflight completed all ten scheduler cells but
accepted no nonzero closed-loop actor update. The guard was applied from an
infeasible anchor: all learned arms began with twenty frequency-endpoint
violations. Many full actor steps preserved every reward floor and reduced the
number or aggregate mass of frequency violations, but increased the worst
remaining endpoint. The feasible-set maintenance ordering therefore rejected
every full and backtracked step and all learned arms selected iteration `-1`.

v14.15 changes the acceptance filter, not the objective, data budget, or actor
update. It separates feasibility restoration from feasibility maintenance.
Discrete violation count determines the phase only; it is not optimized during
restoration because threshold crossings make it discontinuous.

## Frozen Identity

- Core protocol: `freq_hrl_mujoco_shared_core_v14_15_closed_loop_restoration_filter`
- Algorithm revision: `8cc7ecd537167f05c04d9df9792cb5de88c5ce52`
- Source manifest: `f52ffdacf29a0e90567f86ab7cfa4221aa422b890808c0ae1289707e06f207d5`
- Development protocol: `mujoco_v14_15_closed_loop_restoration_filter_screen_v1`
- Evidence role: single-optimizer-seed development preflight, no confidence interval

The preregistration, launcher, merger, analyzer, and this design are frozen
before any v14.15 outcome is accessed.

## Two-Phase Filter

The guard retains two fresh roots crossed with four training disturbances,
giving eight deterministic closed-loop paths. Every candidate is compared with
the frozen anchor on the same paths. The 24 constraints remain four reward
floors and five frequency endpoints for each disturbance.

Let `v_j(theta) >= 0` be the positive normalized violation for frequency
endpoint `j`. The restoration merit is

`M(theta) = sum_j v_j(theta)^2`.

The fixed funnel is `F = gamma * max_j v_j(theta_0)`, where `theta_0` is the
initial continuation policy and `gamma` is preregistered as 2 or 3. For a trial
fraction `alpha` and minimum reduction `eta = 1e-4`, a restoration step is
accepted only when:

1. every reward-floor violation is zero;
2. `M(theta_trial) <= M(theta_current) * (1 - eta * alpha)`;
3. `max_j v_j(theta_trial) <= F`.

Once the current policy has zero frequency violations, the filter switches to
maintenance. Every later accepted trial must keep all reward and frequency
violations at zero. The full step is tested first, followed by fractions
`1/2, 1/4, ...`; rejection restores both actors and their pre-update optimizer
states. Critics and duals retain their on-policy update.

Every full step, backtrack, and rollback is persisted with its fraction,
violation counts, merit, worst violation, decision, and rejection reasons. The
merge independently recomputes the decision rule and rejects missing,
nonmonotone, or internally inconsistent traces.

## Frozen Arms

The preflight contains one shared anchor and nine continuations:

- mean control, function-preserving router calibration, and paired comparator;
- the v14.14 strict closed-loop guard at reward tolerance 0.001 and four
  backtracks;
- restoration filters at reward tolerance 0.001 with funnel 2 or 3;
- restoration filters at reward tolerance 0.005 with funnel 2 or 3;
- one reward-tolerance 0.005, funnel-3 depth check with eight backtracks.

All learned arms retain groupwise projection, frozen-anchor state replay, PPO
trust-region checks, the same dual rates and projection depth, 24 continuation
iterations, and the same checkpoint rule. Only the five restoration arms can
authorize expansion. The strict v14.14 arm is a causal control.

## Preflight Gates

Expansion requires one frozen HalfCheetah replicate to pass all of the following:

- exact source, manifest, seed-role, checkpoint, and scheduler provenance;
- exact function-preserving projection calibration;
- a selected checkpoint at iteration 7 or later;
- at least one nonzero restoration-filter actor update;
- valid trial traces, monotone restoration merit, and strict maintenance after
  feasibility is reached;
- zero selected guard reward and frequency violations;
- zero paired checkpoint-selection violations;
- reward non-inferiority and all five registered frequency reductions for every
  held-out disturbance, including `ood_chirp`;
- nonzero actor-parameter and executed-action changes;
- valid projection, replay, and PPO trust-region diagnostics;
- at least one held-out condition with a strictly positive reward difference.

Failure produces `do_not_expand`. Passing only authorizes a larger multi-seed
development screen. This preflight cannot support performance, robustness,
cross-task generality, statistical significance, or a manuscript claim.

## Scheduler Contract

The formal preflight uses the external scheduler on `node001` through `node006`,
one CPU and 768 MB RAM per task, `require_node=None`, and node-down rerouting.
Continuations wait for the source-bound anchor checkpoint and summary. Analysis
requires terminal scheduler records, run-scoped result synchronization, a
successful frozen merge, and all four required artifacts for every task.
