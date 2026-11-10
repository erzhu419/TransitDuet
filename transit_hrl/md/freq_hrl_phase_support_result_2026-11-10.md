# Phase Support Result

All four tasks t214242-t214245 exited 0; prior-head reproduction and DF8/96-future
budgets passed. Main cost: 5,059.014 CPU seconds, 6,002,400 simulator steps;
retrieved 214,955 public and 2,038 scalar-metadata bytes, no full models/raw banks.
No controller training. Parent decision remains HOLD.

## Complete Episodes

Mean ISE, lower is better; eight new paths/root:

| Task | Start/plain | Start/clock | Checks/plain | Checks/clock | History | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PointMaze | 0.942535 | 0.958170 | 0.996979 | 1.009466 | 0.844297 | 0.961065 |
| Reacher | 4.769864 | 4.727617 | 4.735580 | 4.714542 | 4.661278 | 4.563260 |

Joint versus start/plain changes ISE +7.10%/-1.16%; versus history +19.56%/+1.14%.
Reacher's factorial gains lack CI support. PointMaze root1 sampling with clock
has harmful CI [+0.00423,+0.07005]; its joint variant loses all eight paths.
Reacher's joint gain depends on path1972219007 (-1.17076); the remaining15
average +0.01904. This sensitivity excludes nothing from the primary cohort.

## What Was Resolved

All fitting banks cover five phases; deployment uses later checks with24 calls.
Checks/clock offset weights are nonzero on all roots. Start/clock offset weights
are zero: its changes reflect remaining time/regularization, not learned phase.
The inputs are connected but not a sufficient performance repair.

Late W MSE improves on three roots, but worsens31.6%/56.1% across future halves
on Reacher root1. Local prediction and repeated-policy episode value still differ.

Next: freeze policies; separately vary prefix and continuation policy on matched
states/futures, measuring occupancy and local/full value before another fit.
No holdout retuning or new experiment jobs followed this readout.

Limits: two roots/task, descriptive unadjusted path CIs; additive clock features
and reference-visited coverage. This rejects this repair's sufficiency, not all
phase information. Matching is fitting count/DF, not CPU. Termination competence
and unknown interrupted costs still prevent confirmation.

Sources: [primary summary](../results/plan_validity_phase_support_20261110/summary.json),
[sensitivity](../results/plan_validity_phase_support_20261110/episode_sensitivity.json),
[clock metadata](../results/plan_validity_phase_support_20261110/clock_metadata.json).
