# MuJoCo v14.15 Closed-Loop Restoration-Filter Preflight Outcome

Date: 2026-08-10

## Frozen preflight

- Run: `mujoco_v14_15_closed_loop_restoration_filter_preflight_20260810_r1`
- Scheduler tasks: anchor `t80228`; continuations `t80229..t80237`
- Frozen algorithm revision: `8cc7ecd537167f05c04d9df9792cb5de88c5ce52`
- Frozen source manifest: `f52ffdacf29a0e90567f86ab7cfa4221aa422b890808c0ae1289707e06f207d5`
- Environment: `HalfCheetah-v5`
- Optimizer replicates: one fresh development seed (`1361331598`)
- Held-out paths per continuation: 5 disturbance modes x 8 evaluation seeds
- Evidence role: mechanism preflight only; no optimizer-replicate CI

All ten cells exited naturally with code zero through scheduleurm. The
run-scoped sync admitted exactly one anchor and nine continuations. The frozen
merge validated all required checkpoints, histories, evaluation grids, source
identities, trial traces, restoration phases, merits, funnels, and reward
budgets before analysis.

## Audit erratum

The first merge attempt rejected all five restoration arms because the screen
spec duplicated the production constraint-contract label with semantically
equivalent but byte-different wording. The production core emitted
`infeasible_start_merit_restoration_and_feasible_maintenance_filter_v8`; the
duplicated spec expected
`two_phase_continuous_violation_merit_restoration_and_strict_feasibility_maintenance_v8`.

The post-run verifier was repaired to reconstruct the expected label through
the production core function using the already frozen mechanism flags. The
repair changed no algorithm, result, seed, threshold, preregistration, or
selection rule. Nine protocol tests passed before merge was repeated, and the
corrected merge admitted all ten cells.

## Frozen decision

The independent analyzer returned `expand_to_multiseed_screen`. Four of five
restoration arms passed every authorizing gate. The selected arm was
`group_replay1_trust1_outer1_restore1_eps5e3_bt8_f3`.

For the selected arm:

- the checkpoint selector admitted iteration `7`, meeting the frozen minimum;
- the paired actor RMS difference was `0.002387`, and executed actions changed;
- the closed-loop guard performed 89 evaluations and accepted 22 effective
  joint actor updates;
- the initial 20 frequency violations were reduced to zero;
- continuous violation merit fell from `0.055402` to `0.0`, and the filter
  reached strict feasibility maintenance;
- selected reward and frequency violation counts were both zero;
- anchor replay, PPO trust-region, projection, and selected-checkpoint
  feasibility audits all passed;
- all five held-out disturbance modes passed reward and five-endpoint frequency
  gates, including OOD chirp.

The descriptive mean-return differences versus the matched comparator were
positive in all five modes (`+40.24` to `+156.57`). Every registered frequency
endpoint was reduced in every mode; reductions ranged from `31.48%` to
`45.14%`. These are one-optimizer-seed development observations, not confidence
intervals or confirmatory effects.

One restoration arm (`eps=0.005`, `bt=4`, `funnel=3`) failed selected-checkpoint
feasibility and held-out gates. The strict non-restoration control happened to
reach feasibility on this seed, but it was preregistered as a causal control
and cannot authorize expansion.

## Claim boundary

Allowed: v14.15 demonstrates on one frozen HalfCheetah development seed that
the two-phase restoration filter can admit nontrivial actor updates, eliminate
the registered closed-loop frequency violations under the reward floor, and
produce a candidate that passes the complete preflight gate. This authorizes a
fresh multiseed development screen.

Forbidden: v14.15 preflight alone supports a statistically reliable reward
improvement, robust learned frequency separation, no-tradeoff behavior,
cross-environment generality, confirmatory evidence, or a submission-ready
selected algorithm.
