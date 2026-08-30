# Freq-HRL MuJoCo v14.20 Zeroth-Order Actor Preflight

## Motivation

v14.10 through v14.13 showed that deterministic action-frequency gradients on
fixed state banks do not control the new policy's closed-loop occupancy.
v14.14 through v14.17 added actual closed-loop guards, but their line searches
could only shrink the PPO-proposed actor direction. The v14.17 smoke showed
that even a tiny fraction of that direction increased closed-loop violation
merit. v14.18 and v14.19 then rejected router-only restoration in Walker2d.

The remaining question is whether a reward-safe closed-loop descent direction
exists in a small, auditable actor subspace. This preflight tests that existence
before modifying the trainer.

## Frozen mechanism

- Actor subspace: the final deterministic mean heads of both upper and lower
  actors; exploration `log_std`, critics, and earlier hidden layers are fixed.
- Search: eight fixed antithetic Rademacher directions at parameter RMS
  `1e-6`.
- Gradient: centered rank differences across the 16 antithetic evaluations,
  ordered by reward violations, frequency merit, worst violation, and violation
  count.
- Candidate pool: the 16 direct perturbations plus both orientations of the
  normalized rank-gradient at RMS `1e-8`, `3e-8`, `1e-7`, `3e-7`, and `1e-6`.
- Router strength: unchanged at the v14.17 value `0.5`.

## Honest path split

The first two frozen v14.17 guard roots, crossed with all four disturbance
modes, form eight design paths. The remaining two roots form eight validation
paths and do not enter direction estimation or candidate selection.

A design candidate is eligible only with zero reward violations, at least
`1e-4` relative frequency-merit reduction, and worst violation no greater than
three times the baseline. The selected design candidate is evaluated once on
the validation split under the same gate.

## Preflight matrix and decision

The matrix has one existing anchor seed (`4196455150`) in each of
HalfCheetah-v5, Hopper-v5, and Walker2d-v5. Three independent one-core jobs are
submitted dynamically through `scheduleurm` to `node001-node006`; no node is
required and Slurm is not used.

Only validation support in all three environments authorizes implementation of
an iterative zeroth-order restoration transaction. Any failed environment
rejects this exact subspace/search contract. This post-boundary preflight is
development evidence only and cannot support a manuscript performance claim.
