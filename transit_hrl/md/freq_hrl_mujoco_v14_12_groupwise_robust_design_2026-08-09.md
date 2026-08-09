# MuJoCo v14.12 Groupwise-Robust Development Protocol

Date: 2026-08-09

## Motivation

The source-bound v14.11 preflight established that iterative deterministic
actor-mean projection can reduce pooled same-batch frequency power by
3.87--10.18%, with zero cumulative PPO surrogate-budget violations. It did not
produce an accepted learned checkpoint. A near-candidate improved mean return
and all five pooled frequency endpoints, but failed the registered
worst-condition paired rank.

The mismatch is structural. Four disturbance rollouts were concatenated before
one mean frequency constraint and one mean reward guard were computed, whereas
checkpoint selection applies a separate paired reward floor and five frequency
targets per disturbance mode and ranks the worst violation.

## Core change

v14.12 retains source-rollout identity in every upper and lower trajectory. For
an active level it:

1. computes deterministic deployment-frequency power separately per rollout;
2. computes a paired-anchor relative target separately on the same states;
3. projects the maximum normalized group excess, not pooled mean power;
4. accepts a projection step only when no group exceeds its cumulative
   pre-projection PPO reward-loss allowance;
5. records group count, groups reaching target, and group reward-budget
   violations for every update;
6. records all 24 checkpoint constraints: four modes times reward floor plus
   five effective/raw/latent frequency endpoints.

Checkpoint-selection seeds remain validation-only. Groupwise correction uses
only the four fresh training rollouts collected in that PPO iteration.

## Frozen identity

- Core protocol: `freq_hrl_mujoco_shared_core_v14_12_groupwise_robust_projection`
- Algorithm revision: `de068387e15018762a130b97447dd06af5baeda5`
- Source manifest: `10063fb6b3f9b125ee73ae090b56b9b39ca1a9e400a7750ac3cb218f088dc155`
- Development protocol: `mujoco_v14_12_groupwise_robust_screen_v1`

## Preflight

The preflight is one fresh HalfCheetah-v5 optimizer seed, one source-bound
anchor, and ten continuations:

- three routing/paired controls;
- the matched v14.11 pooled 10%, `k=8`, scale `3/10` learned comparator;
- six groupwise arms covering 5%/10% targets, `k=8`/`k=16`, and actor-anchor
  coefficients 0, 0.01, and 0.05.

An arm can authorize expansion only if it selects iteration 7 or later, changes
actor tensors and executed actions, has zero scalar and group reward-budget
violations, exposes four training groups at both levels, reaches at least one
group target, satisfies reward noninferiority and all five registered frequency
targets in every evaluation disturbance, and has at least one strict reward
improvement. The pooled learned arm is diagnostic and cannot authorize
expansion.

## Compute contract

Tasks use scheduleurm with `require_node=None`, one physical core per cell, and
dynamic eligibility across `node001..node006`. The preflight contains 11 cells,
far below one node's 192-core capacity; node spreading is not itself a gate.
No `jtl110cpu` artifact is admissible.

## Claim boundary

Before audited scheduler output, only the implementation and frozen protocol
exist. Unit tests or a local smoke cannot support learned separation, reward
improvement, no-tradeoff behavior, cross-task generality, or confirmatory
evidence. A passing single-seed preflight may authorize a larger development
screen but cannot support a manuscript performance claim.
