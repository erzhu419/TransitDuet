# Transit Learned Promotion Replan Validation

| variant | reward | wait | raw drift abs | replans | gate | delta reward | delta wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| interval_only | -4.4530 | 4.1275 | 0.008811 | 0.00 | 0.000 | +0.0000 | +0.0000 |
| deterministic_forced | -4.4539 | 4.1257 | 0.008915 | 21.20 | 0.000 | -0.0009 | -0.0018 |
| learned_gate | -4.4563 | 4.1267 | 0.008547 | 16.40 | 0.675 | -0.0034 | -0.0008 |

Paired checks compare `learned_gate` against `interval_only` by eval seed.

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| learned_gate_vs_interval_reward_mean | not_supported | reward_mean | 5 | -0.0034 | -0.0048 | -0.0019 | 0.00 |
| learned_gate_vs_interval_wait_proxy | supported | wait_proxy | 5 | -0.0008 | -0.0012 | -0.0005 | 1.00 |
| learned_gate_vs_interval_RawLowerLFDriftAbs | supported | RawLowerLFDriftAbs | 5 | -0.0003 | -0.0003 | -0.0002 | 1.00 |
| learned_gate_vs_interval_promotion_replan_count | supported | promotion_replan_count | 5 | +16.4000 | +12.6000 | +22.8000 | 1.00 |
