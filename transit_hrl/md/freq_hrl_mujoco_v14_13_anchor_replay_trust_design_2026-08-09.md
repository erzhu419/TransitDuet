# MuJoCo v14.13 Anchor-Replay Trust-Region Development Protocol

Date: 2026-08-09

## Motivation

The source-bound v14.12 preflight preserved four disturbance groups and all
per-group reward budgets, but it admitted no learned checkpoint. Five of six
groupwise arms selected the initial fallback; the remaining arm selected
iteration 3, below the registered learned-checkpoint minimum. The audit showed
two structural defects:

1. unconstrained PPO actor updates increased deployment-frequency excess before
   the post-update projection could recover it; and
2. candidate-only training states did not cover the frozen anchor policy's own
   closed-loop state distribution.

Further dual-rate or projection-step tuning would not identify either defect.

## Core change

v14.13 adds two jointly testable mechanisms to the shared PPO core:

1. **Frozen anchor-state replay.** Iteration-0 deterministic anchor trajectories
   are collected separately on the four training disturbance paths. Frequency
   constraints are evaluated on both current candidate states and frozen anchor
   states, while reward guards remain restricted to the four current training
   groups so incompatible return bootstraps are never pooled.
2. **PPO actor trust region.** Each upper and lower PPO actor update is evaluated
   against the per-group frequency and reward contracts. A violating full step
   is geometrically backtracked; if no feasible nonzero fraction exists, actor
   and optimizer state are restored exactly. Iterative reward-guarded projection
   remains a post-update feasibility correction.

The runner records replay provenance, current/replay group counts, full-step and
accepted-step violations, accepted fractions, optimizer rollback, and
post-projection feasibility for both levels.

## Frozen identity

- Core protocol: `freq_hrl_mujoco_shared_core_v14_13_anchor_replay_trust_region`
- Algorithm revision: `9f98c57572279611823445ee1e908c73833eeace`
- Source manifest: `fbb3b46e0fb05465b35fbbe4e9c0f7c1e8d6260560924bf5a44c0399071be0e5`
- Development protocol: `mujoco_v14_13_anchor_replay_trust_screen_v1`

The frozen source manifest covers `freq_hrl/`; the committed launcher and spec
hashes are additionally written into each run preregistration.

## Preflight design

The preflight uses one fresh HalfCheetah-v5 optimizer seed, one source-bound
anchor, and 12 continuations:

- three controls: un-routed mean control, projection calibration, and the
  compute-matched routed comparator;
- one v14.12 groupwise control without replay or trust;
- one replay-only and one trust-only ablation at reward tolerance 0.01;
- six joint replay-plus-trust arms spanning reward tolerances `1e-8`, `0.001`,
  `0.005`, `0.01`, and diagnostic-only `0.02`, plus `k=8` versus `k=16` at
  tolerance `0.005`.

Only the four joint arms with reward tolerances from `0.001` through `0.01` can
authorize expansion. Replay-only, trust-only, strict `1e-8`, and diagnostic
`0.02` arms identify mechanism boundaries but cannot be selected.

All optimizer, pretraining, selection, continuation, and evaluation seeds are
fresh and mutually disjoint from v14.12. Selection seeds remain validation-only
inside training; the eight development evaluation seeds are used only after a
checkpoint has been selected.

## Authorization gates

An authorizing arm must satisfy all of the following:

1. select iteration 7 or later and change both actor tensors and executed
   action traces;
2. expose eight frequency groups per level: four current-state groups and four
   frozen-anchor-state groups;
3. expose exactly four current-state reward groups and the four-path frozen
   replay provenance;
4. accept at least one nonzero trust-region PPO update at each level, preserve
   frequency feasibility relative to the pre-update state, and incur zero final
   group reward-budget violations;
5. accept at least one corrective projection step at each level, reach at least
   one group target, and incur zero scalar or per-group projection reward-budget
   violations;
6. meet the paired 2% return-noninferiority floor and all five 5% effective,
   raw, and latent frequency-reduction targets in every registered evaluation
   disturbance mode; and
7. improve reward strictly in at least one registered condition.

The exact pathwise projection calibration and all source/provenance/hash gates
must also pass. Ranking uses the worst held-out frequency reduction, with a
smaller reward tolerance as the tie-breaker. No gate may be relaxed after
outcome access.

## Compute contract

The launcher runs from `transit_hrl/` with `PYTHONPATH=.` and submits through
scheduleurm. Every cell requests one physical CPU core, 768 MB RAM,
`require_node=None`, rerouting on node loss, and dynamic eligibility across
`node001..node006`. The 13-cell preflight does not require even node spreading;
eligible work is placed wherever capacity is available.

`jtl110cpu` is quarantined. Its old tasks are scheduler records whose remote
process-tree termination cannot be confirmed while the host is unreachable;
their displayed wall time is not admissible execution evidence. No artifact
from that host may enter this protocol.

## Claim boundary

Unit tests and local smokes establish implementation behavior only. A passing
single-optimizer-seed preflight may authorize a larger development screen; it
cannot support performance, robustness, no-tradeoff, cross-task, statistical,
or confirmatory manuscript claims. A failed preflight is retained as negative
development evidence and must not be bypassed by an unregistered arm.
