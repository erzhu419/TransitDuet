# Multiscale Goal-Control Stage-1 V1 Result

Date: 2026-09-19

## Evidence role

This is a frozen development result for the reoriented Freq-HRL mainline. It
tests whether a causal multiscale representation and a goal-conditioned
hierarchy add value in an identifiable point-mass tracking system. It is not
confirmatory evidence and it does not support a positive Freq-HRL performance
claim.

## Execution and validity

- Scheduler run: `multiscale_goal_stage1_v1_development_20260919_r1`
- Planned cells: 160
- Unique completed cells: 160
- Independent optimizer roots per method-scenario cell: 8
- Held-out evaluation episodes: 1,280
- Unique train/selection/evaluation role seeds: 128
- Parameter-budget ratio: 0.9944 to 1.0018
- Protocol-valid rows: 1,280 of 1,280
- Non-finite values or unexpected artifacts: 0

Four initial attempts failed scheduler terminal diagnosis and were replaced by
successful automatic retries. The final matrix contains exactly one result for
every registered `(scenario, method, optimizer_root)` cell.

## Registered decision

The mainline HRL increment is **not supported**.

| Scenario | Flat representation | HRL on history | Multiscale HRL vs HRL | Multiscale HRL vs flat multiscale |
|---|---|---|---|---|
| clean | inconclusive | inconclusive | inconclusive | contradicted |
| slow target + fast force | inconclusive | inconclusive | inconclusive | contradicted |
| slow signal + fast observation noise | inconclusive | contradicted | mixed | inconclusive |
| band swap, boundary only | inconclusive | contradicted | supported | inconclusive |

For the direct comparison against `flat_multiscale`, the mean episode-return
improvement of `hrl_multiscale` was -2.420 in clean conditions (95% CI
[-4.497, -0.343]), -2.407 under fast dynamics force (95% CI
[-4.528, -0.287]), and -2.342 under fast observation noise (95% CI
[-4.942, 0.257]). Tracking-RMSE contrasts had the same adverse direction.

The raw-history versus full-multiscale flat-policy comparison was inconclusive
in every scenario. Thus the experiment supports neither a general
representation benefit nor a hierarchy-by-multiscale interaction.

## Interpretation

The flat policy nearly saturates this directly observed, dense-reward tracking
task. The fast zero-mean force also changes performance very little because the
point-mass dynamics already attenuate it. A slower upper policy therefore adds
an execution bottleneck without solving a planning problem that the task
actually requires.

This result must not be tuned into a positive claim by changing gates or adding
more roots. It establishes a useful boundary: a causal multiscale transform and
goal hierarchy are not automatically beneficial when the current target is
directly observed and primitive control already solves the task.

## Next gate

Freeze Stage-1 V1 as negative development evidence. The next experiment is a
clean PointMaze gate:

1. verify that ordinary goal-conditioned HRL learns a genuine waypoint task;
2. compare it with a capacity-matched flat policy on fresh paired roots;
3. add multiscale information only after the ordinary HRL baseline succeeds;
4. keep action-spectrum projection, promotion, leakage loss, and responsibility
   gauge disabled.

The generated analysis is stored under
`results/multiscale_goal_stage1_v1_development_20260919_r1/analysis/`.
