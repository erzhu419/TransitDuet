# MuJoCo v14.10 Deployment-Aligned Development Protocol

Date: 2026-08-09

## Motivation

MuJoCo v14.9 optimized frequency costs computed from sampled Gaussian rollout
actions but evaluated deterministic actor-mean actions. The learned constraint
therefore acted mainly on exploration noise and did not reliably improve the
deployed lower-frequency endpoint. v14.10 changes the optimization target
rather than tuning the old dual rates again.

## Frozen core

- Core protocol: `freq_hrl_mujoco_shared_core_v14_10_deployment_aligned_constraints`
- Algorithm revision: `1704ce3f5f867ab493899d424f1398557cc4a625`
- Source manifest: `dbc8cc11399a4baad5cb7c1231860c3911293398f553cd2426670fd08636ea80`
- Development protocol: `mujoco_v14_10_deployment_aligned_screen_v1`
- Evidence role: development only, not confirmatory

The new projected step evaluates the deterministic squashed actor mean on the
same PPO states used by the reward update. It expands upper macro actions by
their exact durations, resets filters at episode boundaries, applies an
8-step high-pass operator to upper actions and a 32-step low-pass operator to
lower actions, and normalizes each violation by its registered target power.

For a learned continuation, the target is

`max((1 - reduction) * paired_anchor_power, absolute_power_floor)`.

The upper and lower multipliers are updated independently. A reward guard may
backtrack or reject a constraint step. The training history records physical
and normalized power before and after the step, target and reference power,
gradient conflict, acceptance, backtracks, and reward-loss change.

## Checkpoint protocol

Every continuation loads the same optimizer-replicate-specific anchor. New
deployment multipliers are reset only after actor, critic, and optimizer state
are loaded, and the reset is recorded in provenance.

Paired-relative checkpoint selection uses only the registered selection paths.
It enforces a 2% reward floor and ranks five frequency endpoints against the
same frozen checkpoint on the same states. Held-out evaluation rows are never
used by selection. An initial-checkpoint fallback is allowed so an unsafe
trained checkpoint is not forced into the result. Such a fallback is not
accepted as learned evidence: formal analysis separately requires a selected
checkpoint from iteration 7 or later, changed actor parameters and actions,
and an accepted deployment-frequency step that reduces physical power.

## Registered development matrix

The full matrix contains 528 cells:

- 3 environments: `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`
- 16 fresh optimizer seeds
- 1 shared anchor plus 10 continuation arms per environment and optimizer seed
- 4 training disturbance modes
- 5 held-out evaluation disturbance modes
- 8 held-out evaluation seeds per optimizer replicate

The 10 continuation arms include a zero-strength mean-reward control, a
function-preserving projection calibration, a same-strength paired zero-dual
control, four joint/asymmetric learned variants, two one-level ablations, and
one stronger 10% relative-target variant.

## Sequential execution gate

The full 528-cell matrix is not launched initially. The first stage is an
11-cell HalfCheetah preflight using one fresh optimizer seed and all registered
arms. It is synchronized and merged as an explicit scoped result.

Expansion requires all of the following descriptive preflight checks:

1. projection calibration preserves paired actor tensors, actions, rewards,
   and traces while improving its registered routed endpoints;
2. at least one learned arm selects iteration 7 or later;
3. its actor and executed-action traces differ from the paired control;
4. every active deployment projection is attempted, accepted, and lowers
   physical frequency power;
5. reward remains within the registered 2% floor in every disturbance mode;
6. all five frequency endpoints meet the arm's registered relative target in
   every disturbance mode;
7. reward has a positive descriptive signal in at least one mode.

Passing this gate only authorizes a multi-seed development screen. It does not
establish a confidence interval or manuscript claim. A full screen must still
pass optimizer-replicate bootstrap gates independently in every environment
and disturbance mode.

## Compute and quarantine boundary

Tasks request one CPU core, 768 MB RAM, and dynamic placement across
`node001` through `node006`. No task is hard-pinned. `jtl110cpu` and
`jtl110cpu2` are excluded. The 81 stale `jtl110cpu` scheduler records remain
quarantined and contribute neither dependencies nor evidence.

## Claim boundary

Allowed before results: v14.10 is a source-bound development protocol that
directly aligns the optimized frequency constraint with deterministic deployed
policy actions and prevents unsafe checkpoint forcing.

Forbidden before results: v14.10 improves reward, establishes joint learned
frequency separation, supports no-tradeoff behavior, generalizes across
MuJoCo environments, or supplies confirmatory manuscript evidence.
