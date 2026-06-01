# Transit Learned Promotion Replan Validation

| variant | reward | wait | raw drift abs | replans | gate | delta reward | delta wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| interval_only | -4.4530 | 4.1275 | 0.008811 | 0.00 | 0.000 | +0.0000 | +0.0000 |
| deterministic_forced | -4.4479 | 4.1197 | 0.008913 | 21.20 | 0.000 | +0.0051 | -0.0078 |
| learned_gate | -4.4505 | 4.1208 | 0.008548 | 16.40 | 0.675 | +0.0025 | -0.0067 |

Paired checks compare `learned_gate` against `interval_only` by eval seed.

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| learned_gate_vs_interval_reward_mean | positive_mixed | reward_mean | 5 | +0.0025 | -0.0001 | +0.0051 | 0.60 |
| learned_gate_vs_interval_wait_proxy | supported | wait_proxy | 5 | -0.0067 | -0.0085 | -0.0053 | 1.00 |
| learned_gate_vs_interval_RawLowerLFDriftAbs | supported | RawLowerLFDriftAbs | 5 | -0.0003 | -0.0003 | -0.0002 | 1.00 |
| learned_gate_vs_interval_promotion_replan_count | supported | promotion_replan_count | 5 | +16.4000 | +12.6000 | +22.8000 | 1.00 |
