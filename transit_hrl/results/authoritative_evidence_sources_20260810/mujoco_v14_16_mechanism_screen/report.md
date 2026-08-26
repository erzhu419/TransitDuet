# MuJoCo v14.16 Crossed Restoration Mechanism Screen

- Status: `primary_mechanism_not_ready`
- Primary arm: `l2_path_freeze_crossreplay`
- Optimizer seeds: `3`
- Statistical unit: optimizer seed; held-out paths are paired only.

| rank | arm | env complete | cell complete | engineering | return | mean margin |
|---:|---|---:|---:|---:|---:|---:|
| 1 | l2_path_trainreplay | 1/3 | 2/9 | 2/9 | 0.004955 | 0.012696 |
| 2 | worst_mode_trainreplay | 0/3 | 4/9 | 4/9 | 0.073637 | 0.041782 |
| 3 | l2_mode_trainreplay | 0/3 | 1/9 | 2/9 | 0.060344 | 0.023933 |
| 4 | l2_path_freeze_crossreplay | 0/3 | 0/9 | 0/9 | 0.001698 | -0.048335 |
| 5 | l2_path_freeze_trainreplay | 0/3 | 0/9 | 0/9 | 0.001936 | -0.054183 |

Three optimizer seeds provide development diagnostics only. Held-out environment paths are paired observations, not independent statistical replicates. Any nominated mechanism requires a frozen larger multiseed development screen followed by fresh confirmation seeds.
