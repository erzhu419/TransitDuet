# Native Transit Promotion Replan Validation

This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.

| variant | seed | reward | wait | cv | score | upper decisions | promotion strength | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| interval_only | 31 | -11870.205 | 4.4900 | 0.2265 | -4.9430 | 66.0 | 1.0000 | 4970 |
| interval_only | 41 | -2919.928 | 5.3360 | 0.6167 | -6.5694 | 66.0 | 0.6250 | 4970 |
| native_promotion_replan | 31 | -10594.317 | 4.9120 | 0.5596 | -6.0312 | 260.0 | 1.0000 | 4970 |
| native_promotion_replan | 41 | -2657.148 | 4.4570 | 0.4375 | -5.3320 | 254.0 | 1.0000 | 4969 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_promotion_replan_vs_interval_ep_reward | supported | ep_reward | 2 | +769.3340 | +262.7800 | +1275.8880 | 1.00 |
| native_promotion_replan_vs_interval_avg_wait_min | positive_mixed | avg_wait_min | 2 | -0.2285 | -0.8790 | +0.4220 | 0.50 |
| native_promotion_replan_vs_interval_score | positive_mixed | score | 2 | +0.0746 | -1.0882 | +1.2374 | 0.50 |
| native_promotion_replan_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 2 | +191.0000 | +188.0000 | +194.0000 | 1.00 |
