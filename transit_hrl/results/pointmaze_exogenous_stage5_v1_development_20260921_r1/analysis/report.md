# PointMaze Exogenous-Control Stage-5 Analysis

Protocol: `pointmaze_exogenous_control_stage5_v1`
Evaluation rows: 256
Independent training replicates: 8
Exogenous ordinary-HRL gate: **not_supported**
Frequency-routing admission: **blocked**

| Method | Tracking success mean [95% CI] | Return | Tracking RMSE |
|---|---:|---:|---:|
| flat_exogenous_history | 0.548 [0.300, 0.796] | 186.960 | 0.577 |
| hrl_exogenous_history | 0.616 [0.424, 0.808] | 191.612 | 0.562 |

HRL final-minus-untrained tracking-success status: **supported**.
HRL final-minus-untrained return status: **supported**.
The optimizer root is the statistical unit; held-out episodes are averaged within root.
