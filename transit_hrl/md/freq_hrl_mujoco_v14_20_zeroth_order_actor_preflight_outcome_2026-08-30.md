# Freq-HRL MuJoCo v14.20 Zeroth-Order Actor Preflight Outcome

## Execution

All three frozen environment cells in
`mujoco_v14_20_zeroth_order_actor_preflight_20260830_r1` completed as scheduler
tasks `t84697` through `t84699`. The preregistration records source revision
`779470b01cafcf5c256ad7c1373ad32171e9cc8c`.

## Frozen decision

The preflight was not supported. Validation support was obtained in two of
three environments, below the required 3/3 gate.

| Environment | Design-eligible candidates | Selected design merit | Validation merit | Validation result |
|---|---:|---:|---:|---|
| HalfCheetah-v5 | 0 | none | none | failed before validation |
| Hopper-v5 | 8 | 0.055344 | 0.055350 | supported |
| Walker2d-v5 | 8 | 0.055330 | 0.055339 | supported |

The common baseline merit was approximately `0.055402`. Hopper and Walker2d
selected the negative ranked-gradient direction at output-head RMS `1e-6`.
Their validation reductions were only approximately 0.094% and 0.113%, and all
20 frequency constraints remained violated. These are valid but weak local
merit reductions, not restored feasibility.

## HalfCheetah failure

HalfCheetah had no reward-safe design candidate. Even ranked-gradient steps at
RMS `1e-8` produced three reward violations and frequency merit between `4.83`
and `16.51`, compared with baseline `0.0554`. Larger steps remained unstable.
The best antithetic candidate by reward count still had one reward violation,
merit `12.99`, and worst normalized frequency violation `2.48`.

This is not evidence that an actor descent direction cannot exist. It shows
that a local objective based on eight exact, 1000-step deterministic paths is
not locally smooth enough in HalfCheetah for this zero-order transaction. Tiny
parameter changes alter the long-horizon state trajectory and dominate both
the reward floor and frequency endpoint comparison.

## Decision boundary

The exact v14.20 output-head subspace/search contract is rejected. Increasing
the number of random directions or shrinking the step below `1e-8` would not
address the identified target instability.

The next admissible mechanism must estimate a distributional closed-loop
objective over a substantially larger independent root ensemble and keep a
separate validation ensemble. Candidate evaluations should be parallelized
across physical cores so the statistical smoothing does not create a serial
runtime bottleneck. This remains development work and does not support a
manuscript performance claim.
