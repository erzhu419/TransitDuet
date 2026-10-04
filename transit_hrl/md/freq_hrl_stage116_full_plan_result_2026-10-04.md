# Freq-HRL Stage116: Complete Explicit Plan-Coordinate Head

Date: 2026-10-04

## Protocol

Stage116 kept the Stage115 basis-5, eight-dimensional explicit Bernstein plan
and local option credit, but made all eight plan coordinates trainable through
one zero-initialized readout. The donor upper actor, lower branch, critic,
standard deviations, environment, paired noise, KL radius, seeds, and
evaluation protocol were unchanged. The initial policy was therefore exactly
the same as the Stage115 zero-residual policy.

## Evidence

- Full run: `pointmaze_upper_full_plan_train_stage116_full_20261004_r1`
- Scheduler: `t135338`–`t135346`, all terminal `done`
- Mechanical gate: passed
- Cost: 8 roots, 9,728 native episodes, 11,673,600 lower steps, 8,192 MC calls, 128 residual updates
- Period 50 learned minus forecast: mean `-0.00007905513101924555`, corrected CI `[-0.0004450989842252895, 0.00015363443009341893]`
- Period 100 learned minus forecast: mean `0.00010018793650079516`, corrected CI `[-0.00015296537211195727, 0.0003836776969441402]`
- All four primary endpoints: `inconclusive`
- `upper_gain_gate`: `not_supported`

## Interpretation

Allowing the complete explicit plan-coordinate head to train did not produce a
reproducible upper-plan gain. Together with Stage113--115, this rules out the
current forecast-anchored upper-residual training path as the next mainline
for the general algorithm. It does not prove that hierarchical control is
useless; it shows that the current PointMaze lower loop does not provide a
confirmed performance opportunity for this upper-plan intervention.

## Next Step

Stop adding upper residual capacity. Run the Stage-8 plan-dependency
qualification first: with the lower policy frozen, compare fresh paired
rollouts under normal plan updates, delayed updates, held plans, and bounded
plan perturbations. Measure the continuous tracking loss and recovery, so the
next algorithm change is conditional on demonstrating that the environment
and lower controller expose a real plan-validity/replanning problem.
