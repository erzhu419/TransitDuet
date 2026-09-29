# Multiscale Goal-Control Stage-1 Analysis

Protocol: `multiscale_goal_control_stage1_v1`
Evaluation rows: 1280
Independent training replicates: 8
Mainline HRL increment: **not_supported**

| Scenario | Flat representation | HRL on history | Multiscale HRL vs HRL | Multiscale HRL vs flat multiscale |
|---|---|---|---|---|
| clean | inconclusive | inconclusive | inconclusive | contradicted |
| slow_target_fast_force | inconclusive | inconclusive | inconclusive | contradicted |
| slow_signal_fast_observation_noise | inconclusive | contradicted | mixed | inconclusive |
| band_swap | inconclusive | contradicted | supported | inconclusive |

Positive differences mean improvement: higher episode return or lower tracking RMSE.
The band-swap row defines the assumption boundary and is not included in the success gate.
