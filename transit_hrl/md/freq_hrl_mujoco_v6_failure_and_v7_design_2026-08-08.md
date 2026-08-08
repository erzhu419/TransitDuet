# Freq-HRL MuJoCo v6 Failure And v7 Design

Date: 2026-08-08

## Decision

MuJoCo protocol v6 passed every software and integrity gate but failed its
performance gate. It is retained as development evidence and must not be used
for a superiority or no-tradeoff claim. No fresh confirmatory seeds are to be
spent on v6.

## Validated v6 Matrix

- source revision: `085952e0e0e6d8696b3ca6e84f6bdbd46b914708`;
- protocol: `freq_hrl_mujoco_shared_core_v6_reward_guarded_projection`;
- preflight: 4/4 valid cells, zero issues, zero warnings;
- pilot: 36/36 valid cells and 900 held-out episode rows, zero issues, zero
  warnings;
- evidence role: `development_only_not_claim_eligible`.

The following values average all five evaluation disturbance conditions after
first aggregating within each of three independent optimizer replicates.

| environment | v6 return delta vs no leakage | v6 drift ratio vs no leakage |
|---|---:|---:|
| HalfCheetah-v5 | -271.57 | 2.30 |
| Hopper-v5 | -7.09 | 0.75 |
| Walker2d-v5 | +5.34 | 0.62 |

Walker2d is a genuine positive no-tradeoff development result. Hopper reduces
the reward tradeoff but weakens leakage suppression. HalfCheetah fails both
objectives and rejects the protocol as a domain-general repair.

## Failure Mechanism

v6 first applied reward-only Adam and then modified parameters manually with a
projected cost correction. The correction passed the same-minibatch reward and
cost surrogate checks approximately 99% of the time. This high acceptance rate
did not imply stable episode performance because:

1. the manual correction was absent from Adam's first- and second-moment state;
2. each later reward step therefore used optimizer moments inconsistent with
   the actual policy parameters;
3. a local on-policy surrogate guard did not cover future state-distribution
   shift;
4. small accepted corrections can change a chaotic locomotion training path
   over many iterations.

The result is useful negative evidence: gradient orthogonality alone is not a
performance guarantee, and it must not be described as one in the paper.

## v7 Update Contract

Protocol `freq_hrl_mujoco_shared_core_v7_reward_guarded_adam_projection`
keeps every candidate update inside the same Adam state transition:

1. snapshot policy parameters and the complete Adam state;
2. compute the ordinary reward-plus-entropy Adam candidate;
3. restore the exact pre-update state;
4. project the leakage gradient against the reward-surrogate gradient;
5. generate constrained Adam candidates with geometric cost scales;
6. compare each constrained candidate with the reward-only candidate, not with
   the stale pre-update policy;
7. accept only when reward surrogate is no worse and leakage surrogate is no
   higher than reward-only; otherwise restore the reward-only parameters and
   Adam state exactly.

This makes the fallback behavior operationally identical to the unconstrained
optimizer step for that minibatch. It still guarantees only a sampled local
surrogate relation. Episode-level claims require a new source-bound pilot and,
if that gate passes, fresh confirmatory optimizer and evaluation seeds.
