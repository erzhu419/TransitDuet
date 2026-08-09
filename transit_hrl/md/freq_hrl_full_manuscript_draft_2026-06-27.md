# Frequency-Separated Hierarchical Reinforcement Learning For Time-Series Control

Draft date: 2026-06-27

> **RETIRED LEGACY SCAFFOLD. Do not submit or quote this file as the current
> evidence state. The authoritative ledger is
> `transit_hrl/evidence/authoritative_registry_v1.json`; a replacement
> manuscript must be generated only from that ledger.**

## Abstract

Many control problems are driven by exogenous time series that mix slow regime structure with fast residual disturbances. We introduce Freq-HRL, a frequency-separated hierarchical reinforcement learning protocol with causal routing, promotion-triggered replanning, and leakage accounting. The v2 evidence policy admits only observed raw outcomes to headline claim status; deterministic outcome projections remain sensitivity analyses. The current package is therefore a research implementation under confirmatory validation, not a completed domain-general performance result.

## 1. Introduction

The paper's core claim is narrow: frequency decomposition is not merely a representation trick; it is a control-responsibility principle for HRL. The low-frequency component should primarily shape plans, the high-frequency component should primarily shape local corrections, and persistent high-frequency evidence should trigger controlled replanning. The manuscript should not drift into a deployment paper or a universal RL-convergence paper.

## Claim Boundary

Allowed claim: Freq-HRL implements and evaluates frequency-responsibility routing under explicit raw-outcome validation boundaries.

Disallowed claims: full same-agency Transit OD/onboard-load deployment validation, production exchange execution, universal encoder dominance, and universal nonconvex actor-critic convergence.

## 2. Method

Freq-HRL consists of a causal spectral encoder, upper planner, lower residual controller, promotion gate, leakage accounting, and paired claim-gating protocol. Domain adapters provide endogenous state and rollout semantics; the core interface remains domain-free.

## 3. Experiments

Experiments are organized around claim boundaries rather than isolated metrics: baseline/ablation evidence, native Transit promotion, public real-demand service response, leakage no-tradeoff, advanced encoders, stress regimes, and order-book replay infrastructure.

## 4. Results

The historical raw-only C1-C9 matrix supported 1 of 9 registered claims at the
time of this draft. This table is retained for provenance and is not the
current paper evidence ledger.

| claim_id | status | allowed_wording |
| --- | --- | --- |
| C1 | partial | The frozen evidence partially supports 'Native learned promotion improves reward and wait'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. |
| C2 | partial | The frozen evidence partially supports 'Native real AFC/APC-profile demand improves observed score/reward and strict wait/alighting/throughput'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. |
| C3 | not_supported | The frozen evidence does not support 'Large-scale venue-grade L2/L3 order-book replay is validated'; report it as an unresolved target. |
| C4 | partial | The frozen evidence partially supports 'Advanced encoder evidence spans Quant and Transit'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. |
| C5 | partial | The frozen evidence partially supports 'Leakage no-tradeoff holds beyond surrogate'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. |
| C6 | partial | The frozen evidence partially supports 'Formal theory appendix covers main protocol claims'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. |
| C7 | not_supported | The frozen evidence does not support 'Native promotion reward/wait improvement replicates across stress regimes'; report it as an unresolved target. |
| C8 | partial | The frozen evidence partially supports 'Strong baseline and ablation table supports frequency-responsibility claim'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. |
| C9 | supported | Synthetic stress coverage is supported for the registered stationary, burst, persistent, and OOD regimes. |

## 5. Discussion And Limitations

The current evidence supports implementation and bounded mechanism claims. Raw native improvement, matched learned baselines, large real replay, and verified theory remain open.

## Figure Plan

Fig. 1: Frequency-separated protocol. Fig. 2: Claim and ablation matrix. Fig. 3: Transit promotion and real-demand service response. Fig. 4: External Transit data coverage. Fig. 5: Order-book replay and encoder generalization. SI: scheduler seeds, data scripts, paired-CI rules, and proof details.
