# Transit Learned Promotion Replan Validation

| variant | reward | wait | raw drift abs | replans | gate | delta reward | delta wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| interval_only | -4.4738 | 4.1237 | 0.008853 | 0.00 | 0.000 | +0.0000 | +0.0000 |
| deterministic_forced | -4.4672 | 4.1148 | 0.008948 | 25.00 | 0.000 | +0.0067 | -0.0089 |
| learned_gate | -4.4692 | 4.1158 | 0.008578 | 20.10 | 0.678 | +0.0047 | -0.0079 |

Paired checks compare `learned_gate` against `interval_only` by eval seed.

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| learned_gate_vs_interval_reward_mean | supported | reward_mean | 10 | +0.0047 | +0.0016 | +0.0076 | 0.70 |
| learned_gate_vs_interval_wait_proxy | supported | wait_proxy | 10 | -0.0079 | -0.0096 | -0.0062 | 1.00 |
| learned_gate_vs_interval_RawLowerLFDriftAbs | supported | RawLowerLFDriftAbs | 10 | -0.0003 | -0.0003 | -0.0003 | 1.00 |
| learned_gate_vs_interval_promotion_replan_count | supported | promotion_replan_count | 10 | +20.1000 | +14.9000 | +25.5000 | 1.00 |
