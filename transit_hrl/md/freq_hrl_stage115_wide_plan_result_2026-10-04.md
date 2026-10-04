# Freq-HRL Stage115: Wider Bernstein Plan Head

Date: 2026-10-04

## Protocol

Stage115 widened the explicit upper Bernstein plan from basis 3 / 4 action
coordinates to basis 5 / 8 action coordinates. The actor output layout was
kept as four donor coordinates followed by four zero-initialized high-order
coordinates; the decoder reordered them to per-entity
`[anchor, donor0, donor1, extra0, extra1]` coefficients. Only the four new
coordinates were trainable. The Stage112 learned lower branch, donor upper
actor, standard deviations, critic, environment, paired noise, local option
credit, KL radius, seed roster, and evaluation protocol were frozen.

## Evidence

- Full run: `pointmaze_upper_wide_plan_train_stage115_full_20261004_r1`
- Scheduler: `t135305`–`t135313`, all terminal `done`
- Mechanical gate: passed
- Cost: 8 roots, 9,728 native episodes, 11,673,600 lower steps, 8,192 MC calls, 128 residual updates
- Period 50 learned minus forecast: mean `-0.0002538729151506658`, corrected CI `[-0.0007188748579869397, 0.00017163305013134789]`
- Period 100 learned minus forecast: mean `0.000007156383359685492`, corrected CI `[-0.000302642612509868, 0.00029709843347092146]`
- All four primary endpoints: `inconclusive`
- `upper_gain_gate`: `not_supported`

## Interpretation

The wider high-order coordinates did not establish an upper-plan performance
gain. The result does not support a claim that adding Bernstein capacity alone
solves the forecast-equivalent upper policy problem.

## Next registered factor

Train the complete 8-dimensional explicit plan-coordinate readout, including
the four donor-coordinate corrections and four high-order coordinates, from a
zero residual initialization. Keep the lower branch and all Stage115
evaluation controls frozen so the next result isolates trainable upper plan
coordinate coverage rather than another change to the environment or credit
protocol.
