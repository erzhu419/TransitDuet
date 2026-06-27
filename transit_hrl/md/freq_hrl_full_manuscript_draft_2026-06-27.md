# Frequency-Separated Hierarchical Reinforcement Learning For Time-Series Control

Draft date: 2026-06-27

## Abstract

Many control problems are driven by exogenous time series that mix slow regime structure with fast residual disturbances. Generic flat policies and generic hierarchical policies can blur these responsibilities: high-level policies overreact to noise, while low-level controllers accumulate local corrections into long-horizon plan drift. We introduce Freq-HRL, a frequency-separated hierarchical reinforcement learning protocol that routes low-frequency trend and forecasts to the upper planner, high-frequency residuals to the lower controller, and persistent residual shocks to a promotion-driven replanning path. Leakage diagnostics penalize upper high-frequency oscillation and lower low-frequency drift. Across the current registered evidence matrix, Freq-HRL is supported against non-frequency, raw-history, misrouted-frequency, no-promotion, and no-leakage alternatives, with native Transit promotion, public AFC/APC demand service-response, conservative leakage no-tradeoff gates, and venue-grade L2/L3 replay paths. We present Freq-HRL as a validated protocol for exogenous time-series HRL, while reserving full deployment-scale Transit and production exchange claims for future same-agency and multi-session external validation.

## 1. Introduction

The paper's core claim is narrow: frequency decomposition is not merely a representation trick; it is a control-responsibility principle for HRL. The low-frequency component should primarily shape plans, the high-frequency component should primarily shape local corrections, and persistent high-frequency evidence should trigger controlled replanning. The manuscript should not drift into a deployment paper or a universal RL-convergence paper.

## Claim Boundary

Allowed claim: frequency-responsibility routing improves hierarchical reinforcement learning for non-stationary time-series control under the registered paired validation boundaries.

Disallowed claims: full same-agency Transit OD/onboard-load deployment validation, production exchange execution, universal encoder dominance, and universal nonconvex actor-critic convergence.

## 2. Method

Freq-HRL consists of a causal spectral encoder, upper planner, lower residual controller, promotion gate, leakage accounting, and paired claim-gating protocol. Domain adapters provide endogenous state and rollout semantics; the core interface remains domain-free.

## 3. Experiments

Experiments are organized around claim boundaries rather than isolated metrics: baseline/ablation evidence, native Transit promotion, public real-demand service response, leakage no-tradeoff, advanced encoders, stress regimes, and order-book replay infrastructure.

## 4. Results

The current conservative claim matrix is fully supported under registered boundaries.

| claim_id | status | allowed_wording |
| --- | --- | --- |
| C1 | supported | Native learned promotion improves reward/wait under the registered native stress artifact. |
| C2 | supported | Native public AFC/APC demand service-response improves score, wait, alighting, and throughput in the current validation loop. |
| C3 | supported | Venue-grade L2/L3 replay infrastructure is supported on the current LOBSTER/NASDAQ TotalView-ITCH symbol sessions. |
| C4 | supported | Advanced encoder paths have cross-domain support under bounded public-market and L3 caveats. |
| C5 | supported | Leakage no-tradeoff is supported where same-domain drift reduction and performance gates both pass. |
| C6 | supported | The formal appendix gives sufficient-condition results for the protocol claims. |
| C7 | supported | Promotion improvement replicates across the current registered persistent and OD-shift stress matrices. |
| C8 | supported | Baseline and ablation evidence supports frequency responsibility over non-frequency and misrouted alternatives. |
| C9 | supported | Stress coverage is supported for the registered stationary, burst, persistent, and OOD regimes. |

## 5. Discussion And Limitations

The evidence supports a domain-general protocol claim, not unrestricted deployment readiness. Remaining carrier-class work is concentrated in same-agency Transit data loops, larger venue-grade market replay, stronger flat SAC/TD3 baselines, and final notation polish for the theory appendix.

## Figure Plan

Fig. 1: Frequency-separated protocol. Fig. 2: Claim and ablation matrix. Fig. 3: Transit promotion and real-demand service response. Fig. 4: External Transit data coverage. Fig. 5: Order-book replay and encoder generalization. SI: scheduler seeds, data scripts, paired-CI rules, and proof details.
