# Freq-HRL v7.1 Pilot Audit And v7.2 Promotion Rebuild

Date: 2026-08-08

## Decision

The v7.1 selective-promotion pilot is valid diagnostic evidence, but no
candidate is eligible for the full HPO or a frozen paper configuration. The
next protocol replaces fixed actor-probability thresholding at evaluation time
with a learned paired-counterfactual advantage critic. Held-out OOD and
promotion-recovery paths remain inaccessible during this redesign.

## Audited v7.1 pilot

- protocol: `full_method_support_only_hpo_v7_1`
- code revision: `7afd538ad1d1629247ac4dcc2b470aacec36a6e0`
- cells: 18 valid of 18 expected
- candidates: 6
- independent optimizer/data replicates: 3
- support regimes: stationary low noise, stationary high noise, localized
  burst, and persistent shift
- training budget: 8 iterations, 3 rollout roots, and 120 steps per episode
- checkpoint and tuning seeds: disjoint
- OOD and promotion-recovery access: not loaded
- source identity: verified
- scheduler runtime on the Linux CPU pool: approximately 6 minutes per cell

All six candidates passed the learning gate, but none passed the registered
mechanism gate. The utility-leading `v71_balanced` candidate promoted on 98.3%
of stationary-low-noise opportunities and 99.2% of stress opportunities. The
most selective candidate, `v71_conservative`, reduced those rates to 8.2% and
33.1%, but only two of three independent training replicates were selective;
the third emitted no deterministic promotions. No v7.1 candidate may therefore
be promoted to the full matrix.

## Failure diagnosis

The paired rollout already estimates the incremental discounted value of
replanning versus continuing the executable old plan. In v7.1 that signal
trained the Bernoulli actor, while deterministic evaluation thresholded the
actor probability. Small optimizer-dependent shifts in the actor intercept
could consequently turn a useful ranking into always-on or always-off control.
The failure is a decision-calibration problem, not evidence that persistent
stress lacks incremental replanning value.

## v7.2 algorithm contract

For promotion state `s_k`, the rollout constructs both executable branches
under the same exogenous return path. Let `G_replan` and `G_continue` be their
discounted plan-value increments, including the registered replan cost. The
paired target is

```text
A_cf(s_k) = promotion_credit_scale * (G_replan - G_continue).
```

A separate critic `A_phi(s_k)` is fitted with a Huber objective on `A_cf`.
The Bernoulli actor remains the stochastic behavior policy used to collect
both promotion actions during training. Frozen deterministic evaluation uses

```text
promote(s_k) = 1[A_phi(s_k) >= tau_adv],
```

where `tau_adv` is selected only on support-validation regimes. This separates
exploration probability from the deployable counterfactual decision boundary.
The advantage critic has its own optimizer and is included in all reported
capacity counts. Fixed-architecture promotion ablations retain its parameters
but disable its behavioral effect.

## Pre-registered pilot gate

A v7.2 candidate advances only when at least 80% of independent training
replicates satisfy every condition below:

1. at least one promotion and one resulting upper replan execute;
2. stationary-low-noise promotion rate is at most 0.25;
3. stress promotion rate is in `[0.02, 0.90]`;
4. stress-minus-low promotion-rate lift is at least 0.02;
5. stress-minus-low predicted-advantage lift is at least `1e-3` in scaled
   advantage units;
6. advantage predictions and paired labels have one-to-one alignment;
7. thresholded advantage decision accuracy is at least 0.55 on frozen support
   validation paths;
8. HF action sensitivity, LF reference, upper residual action, and hard HF
   budget projection are all active.

Utility and positive checkpoint learning remain necessary but cannot override
a failed mechanism gate. The pilot is diagnostic only and cannot emit a frozen
configuration. Full HPO begins only after a jointly eligible candidate exists.

## Seed firewall

v7.2 was designed after inspection of v7.1 support-validation outcomes. It
therefore rotates all development data roles rather than reusing the observed
v7.1 paths:

- training rollout roots: `104729, 104743, 104759`;
- checkpoint-validation seeds: `130003, 130021, 130027`;
- tuning-validation seeds: `150001, 150011, 150013, 150019, 150041`;
- diagnostic-pilot optimizer seeds: `3001, 3011, 3023`;
- final-HPO optimizer seeds: `4001, 4003, 4007, 4013, 4019`.

The final freeze gate rejects any matrix that omits a registered final-HPO
optimizer seed or reuses a diagnostic-pilot optimizer seed. Confirmatory test
seeds remain a third, untouched role and are not present in HPO artifacts.

## Compute provenance boundary

The old v6 records on `jtl110cpu` are excluded. Their scheduler state cannot be
reconciled because SSH key exchange fails before process or result probing, and
a force-cancel could not confirm remote process-tree termination. Their
reported elapsed time is therefore unknown remote state, not measured training
runtime. All v7.2 tasks use scheduler-managed dynamic placement over
`node001` through `node006`, with exact environment preflight and source hashes
recorded before training.

## Paper boundary

Until the v7.2 mechanism gate, full support-only HPO, and untouched
confirmatory evaluation all pass, the valid statement is that Freq-HRL has a
paired-counterfactual promotion-learning implementation under validation. The
project must not claim selective learned promotion or promotion-induced reward
improvement from v7.1.
