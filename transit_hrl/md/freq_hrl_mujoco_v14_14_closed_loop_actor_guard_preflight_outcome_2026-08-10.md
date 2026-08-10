# MuJoCo v14.14 Closed-Loop Actor-Guard Preflight Outcome

Date: 2026-08-10

## Frozen run

- Run: `mujoco_v14_14_closed_loop_actor_guard_preflight_20260810_r1`
- Scheduler tasks: `t79738..t79747`
- Terminal states: 10 done, 0 failed, 0 cancelled
- Placement: anchor on `node003`; continuations on `node003` and `node004`
- Core protocol: `freq_hrl_mujoco_shared_core_v14_14_closed_loop_actor_guard`
- Algorithm revision: `94ad4a1d16d763c36a80c2b58f9c841ede3fb4dd`
- Source manifest: `0649cfbdd481df8b004efa9e03642ec81f232501f24093ae28c7e3c1372c61f7`
- Evidence role: single-optimizer-seed mechanism preflight, no confidence interval

The run-scoped sync admitted all 10 cells. The frozen merge then validated the
source identity, anchor checkpoint provenance, training histories, checkpoints,
and registered evaluation grids before the analyzer accessed outcomes. No
`jtl110cpu` artifact was admitted.

## Decision

- Projection calibration: passed
- Eligible arms: none
- Selected arm: none
- Decision: `do_not_expand`

Both matched v14.13 controls and all four closed-loop candidates selected the
iteration `-1` anchor fallback. Consequently, v14.14 does not authorize a
multiseed screen.

## Mechanism result

The closed-loop callback and transaction executed correctly, but every proposed
actor update was rolled back. The frozen anchor had zero reward-floor violations
and 20 small frequency violations, with rank
`[-0.0526316, -0.0554017, 0.02]`. Full actor steps often corrected most of the
20 frequency constraints while preserving the reward floor, but they made the
worst remaining endpoint violation larger. The v14.14 gate required the
lexicographic worst-condition rank never to worsen, so no nonzero fraction was
admissible.

| arm | guard evaluations | effective updates | full-step frequency violations | zero-reward-violation updates | zero-reward L2 improvements | selected iteration |
|---|---:|---:|---:|---:|---:|---:|
| outer only, eps=0.001, bt=8 | 242 | 0 | 2-16 | 3 | 0 | -1 |
| replay + trust + outer, eps=0.001, bt=4 | 146 | 0 | 2-13 | 16 | 1 | -1 |
| replay + trust + outer, eps=0.001, bt=8 | 242 | 0 | 2-13 | 16 | 1 | -1 |
| replay + trust + outer, eps=0.005, bt=8 | 242 | 0 | 2-14 | 14 | 2 | -1 |

The evaluation counts prove that the intended line search was exercised rather
than bypassed. A `bt=8` arm recorded one initial evaluation, 24 updates with nine
candidate fractions plus an exact rollback check, and one selected-checkpoint
evaluation: `1 + 24 * 10 + 1 = 242`. The `bt=4` count is analogously
`1 + 24 * 6 + 1 = 146`.

## Diagnosis

v14.14 uses a feasible-set maintenance rule at an infeasible starting point.
That rule forbids the normal restoration tradeoff in which aggregate violation
mass decreases while the worst individual constraint temporarily increases.
Increasing the backtrack budget cannot repair this geometry: all smaller points
remain on the same locally adverse direction for the worst endpoint. The result
also explains the long runtime: rejected updates exhaust every registered
closed-loop evaluation before exact rollback.

The next admissible mechanism is a two-phase closed-loop filter:

1. while any frequency constraint is violated, accept only reward-safe steps
   that reduce a continuous restoration merit, subject to a bounded worst-case
   violation funnel;
2. after feasibility is reached, enforce zero reward and frequency violations
   as hard maintenance constraints;
3. persist every trial fraction and merit component so rejection cannot be
   hidden behind aggregate counts.

This is a change to the optimization contract, not another step-size or
backtrack-count sweep.

## Claim boundary

v14.14 is negative development evidence. It proves that the independent
closed-loop guard was invoked, exhausted its frozen backtracking contract, and
restored actor and optimizer state exactly. It does not support a trained
checkpoint, held-out frequency separation, reward improvement, no-tradeoff
behavior, cross-task generality, statistical evidence, confirmatory evidence,
or a submission-ready selected algorithm.
