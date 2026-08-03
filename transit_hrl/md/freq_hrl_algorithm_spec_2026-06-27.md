# Freq-HRL Frozen Algorithm Specification

Date: 2026-06-27

## Definition

Freq-HRL is a frequency-responsibility protocol for hierarchical reinforcement learning in environments driven by non-stationary exogenous time series. A causal encoder decomposes the exogenous stream into low-frequency trend, middle-frequency regime buffer, and high-frequency residual. The upper controller owns slow plan variables; the lower controller owns fast residual correction; promotion transfers persistent high-frequency shocks into upper-level replanning; leakage penalties prevent either layer from acting outside its frequency responsibility.

Machine-checkable contract: `transit_hrl/freq_hrl/core/spec.py`. The carrier package writes `spec_validation.json` so the frozen C1-C9 claim ledger and shared-core path audit can be verified without reading prose.

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
| C1 | partial | The frozen evidence partially supports 'Native learned promotion improves reward and wait'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. | Do not claim learned promotion is universally superior under every deployment stress. |
| C2 | partial | The frozen evidence partially supports 'Native real AFC/APC-profile demand improves observed score/reward and strict wait/alighting/throughput'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. | Do not claim one joint agency APC/AFC/OD/onboard-load control deployment. |
| C3 | not_supported | The frozen evidence does not support 'Large-scale venue-grade L2/L3 order-book replay is validated'; report it as an unresolved target. | Do not claim production-scale exchange execution is solved. |
| C4 | partial | The frozen evidence partially supports 'Advanced encoder evidence spans Quant and Transit'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. | Do not claim every advanced encoder dominates in every domain. |
| C5 | partial | The frozen evidence partially supports 'Leakage no-tradeoff holds beyond surrogate'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. | Do not claim no-tradeoff outside domains passing both drift and performance gates. |
| C6 | partial | The frozen evidence partially supports 'Formal theory appendix covers main protocol claims'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. | Do not claim a universal convergence theorem. |
| C7 | not_supported | The frozen evidence does not support 'Native promotion reward/wait improvement replicates across stress regimes'; report it as an unresolved target. | Do not claim all possible stress regimes are covered. |
| C8 | partial | The frozen evidence partially supports 'Strong baseline and ablation table supports frequency-responsibility claim'; only subchecks explicitly marked supported in the raw-only claim matrix may be stated. | Do not claim frequency features alone are the contribution. |
| C9 | supported | Synthetic stress coverage is supported for the registered stationary, burst, persistent, and OOD regimes. | Do not extrapolate to unregistered stress families. |
