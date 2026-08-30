# MuJoCo v17.3 Causal Audit-Optimal Macro Gauge

Date: 2026-08-31

Status: implemented development mechanism; no empirical claim yet.

## Motivation

The v17.2 smooth-macro gauge preserved reward, executed-action, and latent-policy
traces on all 360 paired paths, but no tested EMA coefficient reduced lower
LPF32 or normalized joint merit in any environment. The failure was structural:
the prior-step EMA target could smooth the upper coordinate while moving its lag
into the lower coordinate. More EMA tuning would not address that objective
mismatch.

## Registered Mechanism

At each upper macro boundary, v17.3:

1. observes the current additive total action without future access;
2. uses persistence of that total as the frozen finite-horizon forecast;
3. constructs the exact affine residuals for causal upper HPF8 and lower LPF32,
   normalized by their registered RMS budgets;
4. solves the resulting convex box-constrained quadratic problem by deterministic
   coordinate descent;
5. freezes the upper responsibility plan for the macro interval;
6. projects each requested upper coordinate onto the component-feasibility
   interval induced by the realized total; and
7. assigns the lower responsibility as the exact additive complement.

The optimization target is

```text
mean((HPF8(upper) / upper_budget)^2)
+ mean((LPF32(total_forecast - upper) / lower_budget)^2).
```

The solver starts from a feasible held-upper plan and retains that plan if the
optimized audit objective is worse. This is an objective-level mechanism, not a
post-hoc cutoff adaptation.

## Causality And Identifiability

The frozen plan, rolling audit histories, and low-pass state are updated from the
additive total and canonical responsibilities only. They do not depend on gauge
strength or on the supplied upper/lower factorization. Therefore:

- strength zero and strength one receive identical policy context;
- the nominal and executed actions are unchanged by the intervention;
- full strength is invariant to additive gauge shifts; and
- the upper and lower responsibilities reconstruct the same total at every step.

The feed-forward policy observes the compact internal state needed by the active
plan: causal total low-pass, current canonical plan value, terminal plan target,
and normalized macro phase. This adds one action-vector block and one scalar over
the v17.2 policy state.

## Development Gates

Before any leakage-active multiseed training, a fresh paired mechanism screen
must satisfy all of the following in HalfCheetah-v5, Hopper-v5, and Walker2d-v5:

- exact reward, executed-action, and latent-policy trace equality between gauge
  strengths zero and one;
- exact explicit transition structure and finite registered metrics;
- router and responsibility reconstruction RMS at most `1e-7`;
- at least 10% reduction in upper HPF8 power;
- at least 10% reduction in lower LPF32 power;
- at least 10% reduction in normalized joint merit; and
- mean component projection rate at most `0.25`.

One fresh optimizer seed is a mechanism preflight only. An arm may advance to
fresh multiseed training only if every environment passes. Seeds and result roots
must be new; v17.2 held-out paths are not reusable for selection.

## Current Verification Boundary

Unit and real-runtime smoke tests establish causality, bounded exact
reconstruction, strength-independent context, expanded-state training, and a
strict synthetic improvement over the v17.2 smooth gauge. They do not establish
MuJoCo performance, cross-environment frequency separation, reward improvement,
leakage no-tradeoff, or publication evidence.
