# Transit Learned Promotion Replan Validation

| variant | reward | wait | raw drift abs | replans | gate | delta reward | delta wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| interval_only | -4.4600 | 4.1273 | 0.008803 | 0.00 | 0.000 | +0.0000 | +0.0000 |
| deterministic_forced | -4.4622 | 4.1261 | 0.008909 | 14.00 | 0.000 | -0.0022 | -0.0012 |
| learned_gate | -4.4644 | 4.1267 | 0.008546 | 13.33 | 0.684 | -0.0044 | -0.0006 |

Paired checks compare `learned_gate` against `interval_only` by eval seed.

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| learned_gate_vs_interval_reward_mean | not_supported | reward_mean | 3 | -0.0044 | -0.0055 | -0.0023 | 0.00 |
| learned_gate_vs_interval_wait_proxy | supported | wait_proxy | 3 | -0.0006 | -0.0008 | -0.0004 | 1.00 |
| learned_gate_vs_interval_RawLowerLFDriftAbs | supported | RawLowerLFDriftAbs | 3 | -0.0003 | -0.0003 | -0.0002 | 1.00 |
| learned_gate_vs_interval_promotion_replan_count | supported | promotion_replan_count | 3 | +13.3333 | +12.0000 | +15.0000 | 1.00 |
