# PointMaze Goal-Control Stage-2 Analysis

Protocol: `pointmaze_goal_control_stage2_v3`
Evaluation rows: 256
Independent training replicates: 8
Ordinary HRL learning gate: **supported**
Multiscale admission: **admitted**
Runtime: gymnasium=1.2.0, gymnasium_robotics=1.4.2, mujoco=3.2.7, numpy=2.1.3, pettingzoo=1.26.1, python=3.10.14, scipy=1.13.1, torch=2.5.1+cu121

| Method | Success mean [95% CI] | Return mean | Final distance mean |
|---|---:|---:|---:|
| flat_goal_ppo | 0.719 [0.579, 0.858] | 144.874 | 0.752 |
| hrl_goal_ppo | 0.789 [0.710, 0.868] | 178.147 | 0.526 |

HRL versus flat paired status: **mixed**.
The statistical unit is the independent optimizer root; evaluation episodes are averaged within root.
The ordinary-HRL gate cleared; a separately registered multiscale factorial experiment is now admitted.
