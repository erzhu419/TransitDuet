# MuJoCo v14.28 mechanism portfolio outcome

## Frozen execution

- Source revision: `cb7e553f76`
- Run: `mujoco_v14_28_mechanism_portfolio_preflight_20260830_r3`
- Scheduler tasks: `t84860`-`t84862`
- Placement: dynamic scheduler placement on `node003`; no required node and no Slurm
- Completion: all three tasks finished without retry or runtime error
- Evidence role: adaptive post-v14.27 preflight, not confirmatory evidence

Runs r1 and r2 are excluded. r1 exposed a path-mismatched fold snapshot bug.
r2 executed stale staged source and reproduced the deleted r1 traceback. Neither
run produced outcome metrics. The r3 launcher staged the anchor plus explicit
`scripts/` and `freq_hrl/` source directories before launch.

## Portfolio result

All three environments passed the frozen critic/direction gate, both design
folds, pooled design selection, and independent 128-path validation.

| Environment | Selected mechanism | Router strength | Eligible candidates | Validation merit reduction | Reward violations | Result |
|---|---|---:|---:|---:|---:|---|
| HalfCheetah-v5 | function-preserving router | 0.7 | 5 | 60.00% | 0 | supported |
| Hopper-v5 | function-preserving router | 0.7 | 6 | 56.25% | 0 | supported |
| Walker2d-v5 | function-preserving router | 0.6 | 5 | 30.55% | 0 | supported |

The validation baseline frequency merit was approximately `0.055402` in every
cell. Candidate merits were `0.022161`, `0.024236`, and `0.038474`. The selected
candidate passed both preregistered 64-path design folds in every environment.
The all-environment gate is therefore **supported** (`3/3`).

## Mechanism interpretation

No actor-restoration candidate was selected. The result does not show that an
actor update improves task reward. It shows that the same environment-agnostic
portfolio can reject an unstable actor transaction and select a conservative
responsibility transaction in all three cells.

In the conservative router path, the executed nominal action remains
`upper_policy + latent_lower`; the router transfers a causal component from the
lower responsibility to the upper responsibility with an equal-and-opposite
decomposition. The selected change therefore targets responsibility leakage,
not the physical control action. Zero validation reward-floor violations are
consistent with this construction.

The candidate did not satisfy every relative frequency endpoint. Validation
retained 8, 11, and 12 nonzero mode-level endpoint violations respectively,
with worst normalized violations `0.0526`, `0.0526`, and `0.0736`. The supported
claim is lower aggregate responsibility-restoration merit under the frozen
funnel, not universal satisfaction of every endpoint.

## Decision

The domain-general mechanism portfolio is accepted for shared-core integration.
The integration must preserve the scientific boundary above and add an explicit
pathwise invariance diagnostic for function-preserving candidates: executed
actions and task rewards must match the frozen baseline to numerical tolerance,
while responsibility merit must improve on design and untouched validation.

This preflight is not a paper-level efficacy result. A confirmatory protocol
still needs multiple optimizer seeds and frozen candidate selection without
reusing v14.28 design or validation roots.
