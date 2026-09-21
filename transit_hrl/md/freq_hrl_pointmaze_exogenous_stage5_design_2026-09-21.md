# PointMaze Exogenous-Control Stage-5 Design

Date: 2026-09-21

## Purpose

Stage-4 V2 showed that causal multiscale masking can improve over raw physical
history, but it did not establish the intended upper/lower routing direction.
That experiment decomposed endogenous state history, so actions changed the
future signal being decomposed. Stage 5 returns to the original Freq-HRL
object: a causal external time series whose realization does not depend on the
agent's actions.

## State Contract

The task separates

```text
z_t = [position, velocity]             endogenous physical state
x_t = [moving target, measured force]  exogenous task stream
```

Both hierarchy levels always receive the current, unfiltered physical state
`z_t`. Only `x_t` is stored in the causal history window. The environment
pre-generates `x_t` from the episode seed; two policies taking different
actions receive exactly the same exogenous prefix for the same seed.

No future target or force value is actor-visible. The current force is measured
before action selection and enters the normalized action before the MuJoCo
transition. This is a measured-disturbance control problem, not hidden-noise
robustness.

## Task

The target moves along the valid U-maze centerline at 1.0 world unit per
second, reflecting at route endpoints. Its round-trip period is 12 seconds.
The agent starts near the target at a seed-selected route vertex.

The two force channels are independently phased sinusoids with empirical RMS
0.12 and period 0.04 seconds. PointMaze control `dt` is 0.01 seconds, the upper
decision period is 0.25 seconds, and the external history window is 0.32
seconds. These values make the target slower than upper replanning and the
force equal to the registered fast-period scale in physical units.

At step `t`, the actor observes `(z_t, x_t)`, requests an action, and the task
executes the clipped sum of requested action and measured force. Reward is
`exp(-distance)` between the post-transition position and the pre-action
target. Tracking success is the fraction of steps within the official
PointMaze success radius 0.45. Every episode keeps the fixed 300-step horizon.

## Stage-5 Substrate Gate

Frequency routing remains disabled. The first experiment compares only:

- `flat_exogenous_history`
- `hrl_exogenous_history`

Both use the same 32-step external history, matched interaction budgets, and
approximately matched trainable parameter counts. The HRL upper policy emits a
relative XY waypoint every 0.25 seconds; the lower policy alone emits physical
actions and receives intrinsic waypoint-progress credit terminated at waypoint
changes.

The gate is supported only if, across eight independent optimizer roots:

1. the lower endpoint of the root-level 95% interval for HRL tracking success
   is at least 0.50; and
2. final HRL improves over its own root- and episode-paired untrained policy in
   both tracking success and dense episode return.

Flat-versus-HRL contrasts are reported but are not required to admit the next
frequency experiment. A preflight is software evidence only.

## Admission Boundary

Only a supported substrate gate permits a new equal-shape frequency-routing
experiment over `x_t`. That later experiment must compare raw history,
all-band, intended slow-upper/fast-lower routing, and swapped routing. Stage 5
itself cannot support a frequency-routing or domain-general Freq-HRL claim.

