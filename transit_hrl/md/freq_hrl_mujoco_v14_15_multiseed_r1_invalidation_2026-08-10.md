# MuJoCo v14.15 Multiseed r1 Invalidation

Date: 2026-08-10

## Status

Run `mujoco_v14_15_restoration_multiseed_development_20260810_r1` is invalid
development evidence and must not be merged, analyzed, entered in the
authoritative evidence registry, or cited in the manuscript.

The terminal scheduler state was:

- 300 continuation tasks completed;
- one continuation task failed (`t80747`);
- 104 still-running continuation tasks were cancelled after invalidation;
- all 45 prerequisite anchor tasks had completed and passed the anchor-only
  merge verifier.

No per-cell reward, frequency endpoint, candidate comparison, aggregate, or
confidence-interval outcome from r1 was inspected before invalidation. Only
scheduler state, resource telemetry, the failing traceback, and prerequisite
artifact validity were inspected.

## Failure

Task `t80747` was the preselected restoration candidate on `Hopper-v5` with
optimizer seed `3799486943`. The restoration guard validator compared a
continuous violation merit, which is a sum of squared normalized violations,
against the unsquared `1e-10` violation tolerance. A valid violation just over
`1e-10` can therefore have merit near `1e-20` and was incorrectly rejected:

```text
ValueError: an infeasible guard snapshot must have positive continuous
frequency violations
```

The scheduler-created retry `t81009` used the same defective source revision
and was cancelled before it could repeat the deterministic failure.

## Corrective action

The validator now uses a squared merit tolerance plus a scale-aware floating
point roundoff allowance. Regression coverage includes both the near-threshold
case and an inconsistent frequency-feasible snapshot. The complete local test
suite passed after the repair (`603/603`).

A replacement run must use a new run name, a new preregistration, and one
source identity for every anchor and continuation cell. No r1 checkpoint or
continuation result may be reused in that replacement run.
