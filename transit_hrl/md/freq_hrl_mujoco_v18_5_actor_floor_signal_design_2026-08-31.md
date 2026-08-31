# MuJoCo v18.5 Causal Actor-Floor Signal Design

## Motivation

V18.4 rejects moving-horizon joint projection: the selected mechanism changed
actions severely while leaving 51 realized component traces infeasible. Earlier
router-only work already established a separate boundary. V17.11 met the upper
budget on all paths but could recover only 62/81 split-recoverable failures;
v17.14 then showed that a small causal actor correction could make 119/120 total
traces feasible within the action trust region. The remaining question is
whether a causal frequency-debt signal can identify where that near-successful
actor adapter needs additional correction.

## Diagnostic

For each unchanged v17.8 total-action trace, the existing causal responsibility
planner executes either an H16 hold or H16 damped-velocity forecast. It exposes
the minimum lower LPF32 power attainable over the forecast under the upper HPF8
budget and physical component bounds. V18.5 records six preregistered causal
path scores derived from this signal:

- mean, 95th percentile, and maximum normalized floor excess;
- maximum causal EMA of normalized floor excess with alpha 0.05;
- mean unnormalized floor-power excess;
- forecast joint-infeasibility rate.

The diagnostic also records prefix upper infeasibility, endpoint powers, and
forecast error. It reads reference feasibility labels only for post hoc ranking.
It does not read the seven v17.12 correction targets, reward, observations,
future actions, checkpoints, or fresh paths.

## Confounding Control

All seven actor-floor paths are Hopper paths. A global rank score could therefore
appear useful merely by distinguishing environments. Every signal is evaluated
both globally and against Hopper reference-feasible paths only. The latter is
the required AUC in the decision rule.

## Decision Rule

A separately frozen FIR debt-feedback screen is allowed only if at least one
preregistered causal score satisfies all three conditions:

- Hopper-conditioned rank AUC at least 0.75;
- at least 6/7 actor-floor paths among the global top 14;
- the unresolved v17.14 Hopper OOD-chirp seed 294864529 path among the global
  top 14.

This rule only decides whether the low-dimensional feedback direction is worth
testing. It does not select feedback gains, authorize fresh paths, or support a
manuscript claim.
