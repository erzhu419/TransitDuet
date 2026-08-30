# MuJoCo v16 gauge-training development outcome

Date: 2026-08-30

Status: `training_time_gauge_preflight_not_supported`

## Decision

The v16 training-time gauge preflight does not justify expansion to a larger
fresh-seed experiment. The frozen matrix trained three capacity-matched arms
under three fresh optimizer seeds in HalfCheetah-v5, Hopper-v5, and
Walker2d-v5. All 27 scheduler tasks completed successfully. The nine paired
environment-by-optimizer-seed analysis cells were evaluated on the same 40
held-out disturbance paths per arm.

| Frozen gate | Passed cells |
|---|---:|
| exact responsibility and router reconstruction | 9 / 9 |
| reward noninferiority | 6 / 9 |
| canonical frequency reduction vs joint-band control | 0 / 9 |
| latent frequency noninferiority vs joint-band control | 4 / 9 |
| latent improvement from primal-dual learning | 5 / 9 |

The support gate required all reconstruction, reward, canonical-frequency,
and latent-noninferiority checks, plus latent improvement in at least two of
three optimizer seeds in every environment. It therefore failed without any
threshold change after outcome access.

## Failure analysis

The full-strength total-action gauge solved the identifiability problem it was
designed to solve: its reported upper and lower responsibilities reconstructed
the additive executed action in all nine paired cells. That property alone did
not produce the desired frequency allocation.

The EMA gauge assigned a causal low-pass of total action to the upper level and
the exact complement to the lower level. In Hopper, for example, one optimizer
replicate reduced mean upper-HF power from 0.00321 under the joint-band control
to 0.00044, but increased mean lower-LF drift from 0.00239 to 0.03549. Similar
tradeoffs occurred in Walker2d. The operator is gauge-invariant, but its
first-order EMA transient is not aligned with the LPF32/HPF8 windows used by
the registered responsibility audit.

The second failure is independent. The
`latent_behavior_feasibility_first` checkpoint rank treats reward only as a
tie-breaker after worst frequency violation. All three HalfCheetah candidate
replicates failed reward noninferiority, even though two reduced latent
frequency merit relative to the gauge-only control. A selector that can choose
an arbitrarily low-return checkpoint is incompatible with the no-tradeoff
claim.

## Claim boundary

Allowed: a causal total-action gauge makes additive responsibility coordinates
identifiable and reconstructing, but this EMA implementation did not satisfy
the preregistered cross-environment frequency and reward gates.

Forbidden: v16 validates training-time raw policy separation, leakage
no-tradeoff, reward improvement, or a confirmatory Freq-HRL algorithm.

The next development protocol must change both causes rather than retune the
same grid: use a causal complementary projection aligned with the registered
LPF32/HPF8 audit, and enforce a reward floor before ranking frequency merit.
