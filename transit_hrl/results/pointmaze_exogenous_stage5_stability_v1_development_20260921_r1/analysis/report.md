# PointMaze Stage-5 Optimization-Stability Screen

This is post-hoc development selection, not paper evidence.

| Arm | Worst-root success | Mean success | Mean return | Eligible |
|---|---:|---:|---:|---:|
| v1_control | 0.442 | 0.452 | 181.523 | False |
| dense_rank | 0.442 | 0.452 | 181.523 | False |
| more_rollouts | 0.955 | 0.962 | 243.744 | True |
| dense_rank_more_rollouts | 0.955 | 0.962 | 243.744 | True |

Selected candidate: `more_rollouts`
Stage-5 V2 freeze: **authorized**
