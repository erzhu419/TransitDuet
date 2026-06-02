# Native Transit Promotion Replan Validation

This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.
All variants use lower HF wait action prior gain `45.0s` so promotion is validated inside the full Freq-HRL lower-control loop.
Each native batch uses `1` shared-PPO replay update(s).
Runner workers: `1`.

| variant | seed | reward | wait | cv | score | upper decisions | launch shift | gate replans | wait replans | shift | gate | promotion strength | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interval_only | 301 | -3065.801 | 6.3210 | 0.5195 | -7.3600 | 66.0 | -34.44 | 0.0 | 0.0 | 0.00 | 0.000 | 0.5385 | 4971 |
| interval_only | 411 | -20649.724 | 16.5690 | 0.5165 | -17.6020 | 66.0 | -44.48 | 0.0 | 0.0 | 0.00 | 0.000 | 0.2308 | 4970 |
| interval_only | 901 | -17958.594 | 35.0040 | 0.7281 | -36.4602 | 66.0 | -44.09 | 0.0 | 0.0 | 0.00 | 0.000 | 0.2308 | 4970 |
| interval_only | 1231 | -11408.063 | 8.6610 | 0.5724 | -9.8058 | 66.0 | -44.49 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_promotion_replan | 301 | -3056.533 | 6.2230 | 0.5567 | -7.3364 | 141.0 | -30.73 | 0.0 | 0.0 | 0.00 | 0.000 | 0.5385 | 4971 |
| native_promotion_replan | 411 | -20614.972 | 16.5090 | 0.3710 | -17.2510 | 193.0 | -44.48 | 0.0 | 0.0 | 0.00 | 0.000 | 0.2308 | 4970 |
| native_promotion_replan | 901 | -18006.230 | 34.9640 | 0.5250 | -36.0140 | 166.0 | -44.08 | 0.0 | 0.0 | 0.00 | 0.000 | 0.2513 | 4970 |
| native_promotion_replan | 1231 | -11828.142 | 9.0860 | 0.5188 | -10.1236 | 108.0 | -44.49 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_learned_gate | 301 | -3047.724 | 6.1940 | 0.5402 | -7.2744 | 66.0 | -32.02 | 2.0 | 0.0 | 0.00 | 0.982 | 0.5385 | 4971 |
| native_learned_gate | 411 | -20669.855 | 16.5670 | 0.4115 | -17.3900 | 66.0 | -44.48 | 2.0 | 0.0 | 0.00 | 0.982 | 0.2308 | 4970 |
| native_learned_gate | 901 | -17945.263 | 35.1110 | 0.6382 | -36.3874 | 66.0 | -44.09 | 2.0 | 0.0 | 0.00 | 0.982 | 0.2308 | 4970 |
| native_learned_gate | 1231 | -11596.082 | 9.1840 | 0.3815 | -9.9470 | 66.0 | -44.49 | 2.0 | 0.0 | 0.00 | 0.982 | 0.0000 | 4971 |
| native_wait_aware_replan | 301 | -3035.078 | 6.0830 | 0.5404 | -7.1638 | 66.0 | -30.37 | 1.0 | 1.0 | -3.88 | 0.982 | 0.5385 | 4971 |
| native_wait_aware_replan | 411 | -20610.997 | 16.5030 | 0.4120 | -17.3270 | 66.0 | -44.48 | 1.0 | 1.0 | -0.63 | 0.982 | 0.2308 | 4970 |
| native_wait_aware_replan | 901 | -17962.590 | 34.9770 | 0.5507 | -36.0784 | 66.0 | -44.09 | 1.0 | 1.0 | -2.00 | 0.982 | 0.2308 | 4970 |
| native_wait_aware_replan | 1231 | -11408.063 | 8.6610 | 0.5724 | -9.8058 | 66.0 | -44.49 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_promotion_replan_vs_interval_ep_reward | not_supported | ep_reward | 4 | -105.9238 | -312.7422 | +22.0100 | 0.50 |
| native_promotion_replan_vs_interval_avg_wait_min | not_supported | avg_wait_min | 4 | +0.0568 | -0.0835 | +0.3038 | 0.75 |
| native_promotion_replan_vs_interval_score | positive_mixed | score | 4 | +0.1258 | -0.1506 | +0.3986 | 0.75 |
| native_promotion_replan_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 4 | +86.0000 | +56.5000 | +114.0000 | 1.00 |
| native_learned_gate_vs_interval_ep_reward | not_supported | ep_reward | 4 | -44.1855 | -137.6815 | +15.7040 | 0.50 |
| native_learned_gate_vs_interval_avg_wait_min | not_supported | avg_wait_min | 4 | +0.1253 | -0.0685 | +0.3918 | 0.50 |
| native_learned_gate_vs_interval_score | positive_mixed | score | 4 | +0.0573 | -0.0845 | +0.1772 | 0.75 |
| native_learned_gate_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 4 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_learned_gate_vs_interval_shared_ppo_gate_replans | supported | shared_ppo_gate_replans | 4 | +2.0000 | +2.0000 | +2.0000 | 1.00 |
| native_learned_gate_vs_interval_ep_reward_noninferiority | inconclusive | ep_reward | 4 | -44.1855 | -137.6815 | +15.7040 | 0.50 |
| native_learned_gate_vs_interval_avg_wait_min_noninferiority | inconclusive | avg_wait_min | 4 | +0.1253 | -0.0685 | +0.3918 | 0.50 |
| native_wait_aware_replan_vs_interval_ep_reward | positive_mixed | ep_reward | 4 | +16.3635 | -1.9980 | +34.7250 | 0.50 |
| native_wait_aware_replan_vs_interval_avg_wait_min | supported | avg_wait_min | 4 | -0.0827 | -0.1852 | -0.0135 | 0.75 |
| native_wait_aware_replan_vs_interval_score | supported | score | 4 | +0.2133 | +0.0687 | +0.3354 | 0.75 |
| native_wait_aware_replan_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 4 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_gate_replans | supported | shared_ppo_gate_replans | 4 | +0.7500 | +0.2500 | +1.0000 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_count | supported | shared_ppo_wait_replan_count | 4 | +0.7500 | +0.2500 | +1.0000 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_pressure_mean | supported | shared_ppo_wait_replan_pressure_mean | 4 | +0.3401 | +0.1094 | +0.5375 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_shift_pressure_mean | supported | shared_ppo_wait_replan_shift_pressure_mean | 4 | +0.2034 | +0.0394 | +0.3834 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_gap_ratio_mean | supported | shared_ppo_wait_replan_gap_ratio_mean | 4 | +0.8168 | +0.2660 | +1.1524 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_same_hold_mean | positive_mixed | shared_ppo_wait_replan_same_hold_mean | 4 | +0.3533 | +0.0000 | +0.7065 | 0.50 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_same_wait_mean | supported | shared_ppo_wait_replan_same_wait_mean | 4 | +0.7583 | +0.2621 | +1.1475 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_shift_abs_mean_s | supported | shared_ppo_wait_replan_shift_abs_mean_s | 4 | +1.6273 | +0.3153 | +3.0669 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_shift_mean_s | supported | shared_ppo_wait_replan_shift_mean_s | 4 | -1.6273 | -3.0669 | -0.3153 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_actor_base_used_mean | supported | shared_ppo_wait_replan_actor_base_used_mean | 4 | +0.7500 | +0.2500 | +1.0000 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_base_delta_abs_mean_s | supported | shared_ppo_wait_replan_base_delta_abs_mean_s | 4 | +2.2646 | +0.8006 | +3.3956 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_final_delta_abs_mean_s | supported | shared_ppo_wait_replan_final_delta_abs_mean_s | 4 | +2.2589 | +0.6472 | +3.6976 | 0.75 |
| native_wait_aware_replan_vs_interval_upper_plan_target_mean | not_supported | upper_plan_target_mean | 4 | +0.1138 | -0.7241 | +0.9217 | 0.25 |
| native_wait_aware_replan_vs_interval_terminal_launch_shift_mean | not_supported | terminal_launch_shift_mean | 4 | +1.0181 | +0.0000 | +3.0544 | 0.00 |
| native_wait_aware_replan_vs_interval_ep_reward_noninferiority | supported | ep_reward | 4 | +16.3635 | -1.9980 | +34.7250 | 0.50 |
| native_wait_aware_replan_vs_interval_avg_wait_min_noninferiority | supported | avg_wait_min | 4 | -0.0827 | -0.1852 | -0.0135 | 0.75 |
