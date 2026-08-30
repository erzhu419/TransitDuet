# MuJoCo v15 raw-policy distillation development outcome

Date: 2026-08-30

Status: `universal_raw_policy_distillation_not_supported`

## Decision

The v15, v15.1, and v15.2 development sequence does not justify expansion to
fresh optimizer-seed confirmation. All three protocols reused one frozen v14.29
optimizer seed and used disjoint trajectory roots for distillation, design, and
validation. They are mechanism development, not confirmatory evidence.

| Protocol | Main change | HalfCheetah-v5 | Hopper-v5 | Walker2d-v5 |
|---|---|---|---|---|
| v15 | causal output-head distillation | validation reward floor failed | supported | no design candidate |
| v15.1 | bounded logits and parameter trust region | no design candidate | supported | no design candidate |
| v15.2 | total-action or upper-action causal teacher | no design candidate | supported | no design candidate |

The latest v15.2 run evaluated 216 preregistered candidates. Hopper admitted 83
design-eligible candidates and its selected candidate passed the disjoint
validation roots. HalfCheetah and Walker2d admitted none.

## Failure analysis

HalfCheetah's best candidate removed the upper-HF violations but retained a
normalized raw lower-LF violation of 0.108 and a reward-floor violation of
0.030. Increasing the shared actor-head trust radius worsened lower
compensation and reward.

Walker2d's best candidate met the reward floor and all lower-frequency gates.
Increasing the upper-head trust radius from 0.02 to 0.10 reduced its worst
upper-frequency violation, but the raw upper-HF endpoint still had normalized
violation 0.0529. The alternative upper-action teacher traded that failure for
lower-frequency violations.

These are different failure modes, so a larger version of the same shared
output-head grid is not warranted. The decomposition of a total action into
upper and lower raw outputs has the gauge freedom
`(u, l) -> (u + g, l - g)`. Without an architectural gauge constraint, a
functionally equivalent hierarchy does not have an identifiable raw
factorization. The v15 sequence attempted to pick a factorization after
training; its lack of cross-environment support is consistent with that
non-identifiability.

## Claim boundary

Allowed: post-hoc causal raw-policy distillation worked in Hopper under one
development optimizer seed but was not universal across the three registered
MuJoCo tasks.

Forbidden: v15, v15.1, or v15.2 supports universal raw behavioral separation,
fresh-seed generalization, reward improvement, or a confirmatory Freq-HRL
algorithm.

The next algorithmic line must either impose a gauge-fixed responsibility layer
during training or retain responsibility-space restoration as the explicit
estimand. It must not tune further candidates on optimizer seed `2978317753`.
