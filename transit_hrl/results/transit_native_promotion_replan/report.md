# Native Transit Promotion Replan Validation

This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.

| variant | seed | reward | wait | cv | score | upper decisions | promotion strength | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| interval_only | 31 | -7982.178 | 5.5650 | 0.5159 | -6.5968 | 66.0 | 1.0000 | 4970 |
| interval_only | 41 | -14560.953 | 6.9250 | 0.2622 | -7.4494 | 66.0 | 1.0000 | 4970 |
| interval_only | 51 | -8887.802 | 9.8580 | 0.4928 | -10.8436 | 66.0 | 0.0000 | 4970 |
| interval_only | 61 | -19669.997 | 43.4450 | 0.4684 | -44.3818 | 66.0 | 0.0000 | 4971 |
| interval_only | 71 | -2999.204 | 9.3010 | 0.5432 | -10.3874 | 66.0 | 0.0000 | 4970 |
| interval_only | 81 | -11203.777 | 5.4160 | 0.6081 | -6.6322 | 66.0 | 0.0000 | 4970 |
| interval_only | 91 | -3569.156 | 4.1610 | 0.4257 | -5.0124 | 66.0 | 0.0000 | 4971 |
| interval_only | 101 | -2870.473 | 4.5490 | 0.4948 | -5.5386 | 66.0 | 0.0000 | 4969 |
| native_promotion_replan | 31 | -11333.611 | 5.9310 | 0.5411 | -7.0132 | 157.0 | 1.0000 | 4970 |
| native_promotion_replan | 41 | -3908.690 | 6.3110 | 0.4214 | -7.1538 | 125.0 | 0.0000 | 4971 |
| native_promotion_replan | 51 | -18380.187 | 8.1510 | 0.6657 | -9.4824 | 130.0 | 0.5926 | 4970 |
| native_promotion_replan | 61 | -17097.856 | 10.2280 | 0.3888 | -11.0056 | 134.0 | 0.0000 | 4971 |
| native_promotion_replan | 71 | -2701.025 | 4.6100 | 0.5484 | -5.7068 | 93.0 | 0.0769 | 4970 |
| native_promotion_replan | 81 | -2727.602 | 4.8230 | 0.3653 | -5.5536 | 110.0 | 0.0000 | 4970 |
| native_promotion_replan | 91 | -10199.688 | 4.8180 | 0.5093 | -5.8366 | 84.0 | 0.0000 | 4971 |
| native_promotion_replan | 101 | -2739.135 | 8.5100 | 0.3963 | -9.3026 | 134.0 | 0.6923 | 4970 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_promotion_replan_vs_interval_ep_reward | positive_mixed | ep_reward | 8 | +331.9682 | -4100.6044 | +4771.6282 | 0.62 |
| native_promotion_replan_vs_interval_avg_wait_min | positive_mixed | avg_wait_min | 8 | -4.4798 | -13.1886 | +0.8364 | 0.62 |
| native_promotion_replan_vs_interval_score | positive_mixed | score | 8 | +4.4734 | -0.8151 | +13.2077 | 0.62 |
| native_promotion_replan_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 8 | +54.8750 | +39.5000 | +69.8844 | 1.00 |
