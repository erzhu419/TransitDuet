# Freq-HRL Frozen Algorithm Specification

Date: 2026-06-27

## Definition

Freq-HRL is a frequency-responsibility protocol for hierarchical reinforcement learning in environments driven by non-stationary exogenous time series. A causal encoder decomposes the exogenous stream into low-frequency trend, middle-frequency regime buffer, and high-frequency residual. The upper controller owns slow plan variables; the lower controller owns fast residual correction; promotion transfers persistent high-frequency shocks into upper-level replanning; leakage penalties prevent either layer from acting outside its frequency responsibility.

## Frozen Interface

| component | required contract |
|---|---|
| Exogenous stream | time-stamped, causal, no future observations |
| Causal encoder | emits x_low, x_mid, x_high, low forecast, uncertainty, high energy, persistence |
| Upper policy | consumes low trend, low forecast, uncertainty, promotion signal, leakage feedback, and endogenous plan state |
| Lower policy | consumes active upper plan, local endogenous state, high residual, middle buffer, and shock age |
| Promotion gate | fires only on persistent residual evidence and records false-positive/false-negative boundary |
| Leakage accounting | measures upper high-frequency power and lower low-frequency drift |
| Claim gate | paired, direction-aware CI or explicit noninferiority rule |

## Non-Negotiable Invariants

1. The encoder must be causal: no x_{t+1:T} can enter a policy decision at time t.
2. Frequency features are not sufficient. The experiment must test routing responsibility, not only richer observations.
3. Promotion is a replanning mechanism, not a free improvement label.
4. No-tradeoff is domain-local: drift reduction and performance noninferiority must pass in the same domain.
5. Domain-general means shared core plus domain adapters, not copy-pasted domain algorithms.

## Frozen Claims

| claim_id | status | allowed_wording | disallowed_wording |
| --- | --- | --- | --- |
| C1 | supported | Native learned promotion improves reward/wait under the registered native stress artifact. | Do not claim learned promotion is universally superior under every deployment stress. |
| C2 | supported | Native public AFC/APC demand service-response improves score, wait, alighting, and throughput in the current validation loop. | Do not claim one joint agency APC/AFC/OD/onboard-load control deployment. |
| C3 | supported | Venue-grade L2/L3 replay infrastructure is supported on the current LOBSTER/NASDAQ TotalView-ITCH symbol sessions. | Do not claim production-scale exchange execution is solved. |
| C4 | supported | Advanced encoder paths have cross-domain support under bounded public-market and L3 caveats. | Do not claim every advanced encoder dominates in every domain. |
| C5 | supported | Leakage no-tradeoff is supported where same-domain drift reduction and performance gates both pass. | Do not claim no-tradeoff outside domains passing both drift and performance gates. |
| C6 | supported | The formal appendix gives sufficient-condition results for the protocol claims. | Do not claim a universal convergence theorem. |
| C7 | supported | Promotion improvement replicates across the current registered persistent and OD-shift stress matrices. | Do not claim all possible stress regimes are covered. |
| C8 | supported | Baseline and ablation evidence supports frequency responsibility over non-frequency and misrouted alternatives. | Do not claim frequency features alone are the contribution. |
| C9 | supported | Stress coverage is supported for the registered stationary, burst, persistent, and OOD regimes. | Do not extrapolate to unregistered stress families. |
