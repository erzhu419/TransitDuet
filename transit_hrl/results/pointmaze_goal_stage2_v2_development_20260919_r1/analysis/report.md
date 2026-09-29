# PointMaze Goal-Control Stage-2 Analysis

Protocol: `pointmaze_goal_control_stage2_v2`
Evaluation rows: 256
Independent training replicates: 8
Ordinary HRL learning gate: **not_supported**
Multiscale admission: **blocked**
Runtime: gymnasium=1.2.0, gymnasium_robotics=1.4.2, mujoco=3.2.7, numpy=2.1.3, pettingzoo=1.26.1, python=3.10.14, scipy=1.13.1, torch=2.5.1+cu121

| Method | Success mean [95% CI] | Return mean | Final distance mean |
|---|---:|---:|---:|
| flat_goal_ppo | 0.297 [0.183, 0.411] | 79.432 | 0.946 |
| hrl_goal_ppo | 0.297 [0.152, 0.441] | 75.695 | 0.954 |

HRL versus flat paired status: **inconclusive**.
The statistical unit is the independent optimizer root; evaluation episodes are averaged within root.
Multiscale mechanisms remain blocked unless the ordinary HRL success-rate CI clears the frozen gate.
