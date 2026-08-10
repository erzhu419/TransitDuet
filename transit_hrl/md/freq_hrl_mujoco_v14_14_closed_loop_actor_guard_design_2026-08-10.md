# MuJoCo v14.14 Closed-Loop Actor-Guard Development Protocol

## Motivation

The frozen v14.13 preflight completed successfully as an engineering run but
selected no learned checkpoint. Its inner PPO trust region, anchor-state replay,
and iterative projection constrained deterministic actions on sampled states.
They did not control the state distribution induced by the updated policy on
unseen closed-loop trajectories. This is a distribution-shift failure, not a
scheduler or accounting failure.

v14.14 tests one change: an outer transaction around each joint upper/lower
actor update. The transaction evaluates actual deterministic environment
rollouts on an independent training-constraint registry. It accepts the full
step, halves both actor deltas together, or restores both actors exactly.
Critic and dual updates are retained because they were fit to the pre-update
on-policy batch. A partial or rejected actor transaction restores the pre-update
actor optimizer states.

## Frozen Identity

- Core protocol: `freq_hrl_mujoco_shared_core_v14_14_closed_loop_actor_guard`
- Algorithm revision: `94ad4a1d16d763c36a80c2b58f9c841ede3fb4dd`
- Source manifest: `0649cfbdd481df8b004efa9e03642ec81f232501f24093ae28c7e3c1372c61f7`
- Development protocol: `mujoco_v14_14_closed_loop_actor_guard_screen_v1`
- Evidence role: development preflight, not confirmatory evidence

## Guard Contract

Two fresh guard roots are crossed with the four training disturbances, giving
eight deterministic closed-loop paths. These roots are disjoint from training,
checkpoint-selection, and held-out roots. They are part of training because
they decide whether an actor update is installed; they are never reported as
held-out evidence.

Each evaluation compares the candidate policy with the frozen anchor on the
same paths. The registered constraints are:

1. one 2% reward non-inferiority floor per disturbance;
2. three lower-frequency endpoints per disturbance;
3. two upper high-frequency endpoints per disturbance.

This gives 24 constraints. An actor transaction is accepted only if reward
violations remain zero, the number of frequency violations does not increase,
and the lexicographic tuple `(negative worst violation, negative violation L2,
worst reward-floor slack)` does not worsen. The selected checkpoint is evaluated
again after checkpoint restoration and its violation counts are persisted.

## Causal Arms

The screen retains the frozen anchor, mean control, function-preserving router
calibration, and paired comparator. Six learned continuations separate:

- v14.13 joint replay/trust controls at reward budgets 0.001 and 0.005;
- an outer-guard-only arm;
- joint replay/trust plus outer guard with 4 or 8 backtracks;
- a joint outer-guard arm with reward budget 0.005.

Only the three joint replay/trust plus outer-guard arms can authorize expansion.
The other cells diagnose attribution and cannot be selected regardless of score.

## Preflight Gates

Expansion requires all of the following for one frozen HalfCheetah replicate:

- exact source, manifest, checkpoint, seed-role, and scheduler provenance;
- successful function-preserving calibration;
- a selected learned checkpoint at iteration 7 or later;
- at least one nonzero outer-guard actor update;
- monotone guard history and zero selected guard reward/frequency violations;
- zero paired selection-path violations across reward and all five endpoints;
- reward non-inferiority and all five registered frequency reductions on every
  held-out disturbance, including `ood_chirp`;
- nonzero parameter and executed-action change;
- valid replay, inner trust-region, and projection diagnostics;
- at least one held-out condition with strictly positive reward difference.

Failure produces `do_not_expand`. Passing produces only authorization for a
larger multi-seed development screen; it does not establish performance,
robustness, statistical significance, or a paper claim.

## Scheduler Contract

Tasks use the external scheduler on `node001` through `node006`, request one CPU
and 768 MB RAM per cell, set `require_node=None`, and allow rerouting on node
failure. Continuations wait for the source-bound anchor checkpoint and summary.
The merge requires terminal scheduler records, run-scoped result synchronization,
all four cell artifacts, and exact frozen identity before analysis can run.
