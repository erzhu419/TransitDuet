# Native Transit Promotion Replan Validation

This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.
All variants use lower HF wait action prior gain `45.0s` so promotion is validated inside the full Freq-HRL lower-control loop.
Each native batch uses `1` shared-PPO replay update(s).
Runner workers: `16`.

| variant | seed | reward | wait | cv | score | upper decisions | launch shift | gate replans | wait replans | shift | gate | promotion strength | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interval_only | 241 | -2798.204 | 8.5290 | 0.4780 | -9.4850 | 66.0 | -44.39 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| interval_only | 291 | -11419.763 | 5.4930 | 0.6386 | -6.7702 | 66.0 | -44.43 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| interval_only | 301 | -3065.801 | 6.3210 | 0.5195 | -7.3600 | 66.0 | -34.44 | 0.0 | 0.0 | 0.00 | 0.000 | 0.5385 | 4971 |
| interval_only | 361 | -3024.825 | 5.6790 | 0.3388 | -6.3566 | 66.0 | -44.47 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 371 | -2620.089 | 8.4550 | 0.3905 | -9.2360 | 66.0 | -44.25 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 411 | -20649.724 | 16.5690 | 0.5165 | -17.6020 | 66.0 | -44.48 | 0.0 | 0.0 | 0.00 | 0.000 | 0.2308 | 4970 |
| interval_only | 431 | -9106.576 | 5.0400 | 0.4311 | -5.9022 | 66.0 | -44.50 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4972 |
| interval_only | 551 | -2702.971 | 6.5650 | 0.4744 | -7.5138 | 66.0 | -44.05 | 0.0 | 0.0 | 0.00 | 0.000 | 0.9733 | 4970 |
| interval_only | 561 | -9312.926 | 4.8910 | 0.3895 | -5.6700 | 66.0 | -44.47 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| interval_only | 881 | -2684.396 | 4.1510 | 0.5459 | -5.2428 | 66.0 | -44.16 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4971 |
| interval_only | 901 | -17958.594 | 35.0040 | 0.7281 | -36.4602 | 66.0 | -44.09 | 0.0 | 0.0 | 0.00 | 0.000 | 0.2308 | 4970 |
| interval_only | 971 | -8864.038 | 5.9340 | 0.5509 | -7.0358 | 66.0 | -43.77 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| interval_only | 981 | -11998.161 | 5.4460 | 0.3972 | -6.2404 | 66.0 | -44.17 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| interval_only | 1221 | -2892.566 | 6.3860 | 0.3663 | -7.1186 | 66.0 | -44.44 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4971 |
| interval_only | 1251 | -11930.187 | 5.1120 | 0.1907 | -5.4934 | 66.0 | -44.37 | 0.0 | 0.0 | 0.00 | 0.000 | 0.3846 | 4971 |
| interval_only | 1401 | -2778.479 | 4.6170 | 0.4471 | -5.5112 | 66.0 | -44.41 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 241 | -2793.096 | 8.5280 | 0.4780 | -9.4840 | 123.0 | -44.39 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| native_promotion_replan | 291 | -11462.481 | 5.4860 | 0.4934 | -6.4728 | 127.0 | -44.40 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_promotion_replan | 301 | -3056.533 | 6.2230 | 0.5567 | -7.3364 | 141.0 | -30.73 | 0.0 | 0.0 | 0.00 | 0.000 | 0.5385 | 4971 |
| native_promotion_replan | 361 | -3021.975 | 5.6790 | 0.3388 | -6.3566 | 116.0 | -44.47 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 371 | -2619.211 | 8.4550 | 0.3905 | -9.2360 | 101.0 | -44.25 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 411 | -20614.972 | 16.5090 | 0.3710 | -17.2510 | 193.0 | -44.48 | 0.0 | 0.0 | 0.00 | 0.000 | 0.2308 | 4970 |
| native_promotion_replan | 431 | -9149.147 | 5.0510 | 0.3815 | -5.8140 | 189.0 | -44.51 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4972 |
| native_promotion_replan | 551 | -2701.862 | 6.5730 | 0.4615 | -7.4960 | 102.0 | -43.70 | 0.0 | 0.0 | 0.00 | 0.000 | 0.9733 | 4970 |
| native_promotion_replan | 561 | -9308.197 | 4.8930 | 0.3908 | -5.6746 | 206.0 | -44.46 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| native_promotion_replan | 881 | -2678.419 | 4.1510 | 0.5459 | -5.2428 | 170.0 | -44.16 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4971 |
| native_promotion_replan | 901 | -18006.230 | 34.9640 | 0.5250 | -36.0140 | 166.0 | -44.08 | 0.0 | 0.0 | 0.00 | 0.000 | 0.2513 | 4970 |
| native_promotion_replan | 971 | -8913.529 | 5.9360 | 0.5510 | -7.0380 | 135.0 | -43.76 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_promotion_replan | 981 | -12033.236 | 5.4500 | 0.3969 | -6.2438 | 141.0 | -44.12 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| native_promotion_replan | 1221 | -2906.779 | 6.3860 | 0.3663 | -7.1186 | 134.0 | -44.44 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4971 |
| native_promotion_replan | 1251 | -11820.509 | 5.1140 | 0.1855 | -5.4850 | 192.0 | -44.31 | 0.0 | 0.0 | 0.00 | 0.000 | 0.3846 | 4971 |
| native_promotion_replan | 1401 | -2785.493 | 4.6170 | 0.4471 | -5.5112 | 147.0 | -44.41 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_learned_gate | 241 | -2795.758 | 8.5280 | 0.4779 | -9.4838 | 66.0 | -44.39 | 2.0 | 0.0 | 0.00 | 0.982 | 1.0000 | 4970 |
| native_learned_gate | 291 | -11423.600 | 5.4750 | 0.4932 | -6.4614 | 66.0 | -44.43 | 2.0 | 0.0 | 0.00 | 0.982 | 0.0000 | 4971 |
| native_learned_gate | 301 | -3047.724 | 6.1940 | 0.5402 | -7.2744 | 66.0 | -32.02 | 2.0 | 0.0 | 0.00 | 0.982 | 0.5385 | 4971 |
| native_learned_gate | 361 | -3028.322 | 5.6790 | 0.3388 | -6.3566 | 66.0 | -44.47 | 2.0 | 0.0 | 0.00 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 371 | -2622.083 | 8.4550 | 0.3905 | -9.2360 | 66.0 | -44.25 | 2.0 | 0.0 | 0.00 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 411 | -20669.855 | 16.5670 | 0.4115 | -17.3900 | 66.0 | -44.48 | 2.0 | 0.0 | 0.00 | 0.982 | 0.2308 | 4970 |
| native_learned_gate | 431 | -9109.401 | 5.0420 | 0.4311 | -5.9042 | 66.0 | -44.51 | 2.0 | 0.0 | 0.00 | 0.982 | 1.0000 | 4972 |
| native_learned_gate | 551 | -2711.034 | 6.5690 | 0.4923 | -7.5536 | 66.0 | -44.03 | 2.0 | 0.0 | 0.00 | 0.982 | 0.9733 | 4970 |
| native_learned_gate | 561 | -9305.636 | 4.8890 | 0.4071 | -5.7032 | 66.0 | -44.47 | 2.0 | 0.0 | 0.00 | 0.982 | 1.0000 | 4970 |
| native_learned_gate | 881 | -2689.537 | 4.1510 | 0.5459 | -5.2428 | 66.0 | -44.16 | 2.0 | 0.0 | 0.00 | 0.982 | 1.0000 | 4971 |
| native_learned_gate | 901 | -17945.263 | 35.1110 | 0.6382 | -36.3874 | 66.0 | -44.09 | 2.0 | 0.0 | 0.00 | 0.982 | 0.2308 | 4970 |
| native_learned_gate | 971 | -8902.727 | 5.9470 | 0.6336 | -7.2142 | 66.0 | -43.66 | 2.0 | 0.0 | 0.00 | 0.982 | 0.0000 | 4971 |
| native_learned_gate | 981 | -12010.871 | 5.4480 | 0.3970 | -6.2420 | 66.0 | -44.17 | 2.0 | 0.0 | 0.00 | 0.982 | 1.0000 | 4970 |
| native_learned_gate | 1221 | -2930.166 | 6.3900 | 0.3663 | -7.1226 | 66.0 | -44.44 | 2.0 | 0.0 | 0.00 | 0.982 | 1.0000 | 4971 |
| native_learned_gate | 1251 | -11864.893 | 5.1150 | 0.1855 | -5.4860 | 66.0 | -44.37 | 2.0 | 0.0 | 0.00 | 0.982 | 0.3846 | 4971 |
| native_learned_gate | 1401 | -2786.943 | 4.6170 | 0.4471 | -5.5112 | 66.0 | -44.41 | 2.0 | 0.0 | 0.00 | 0.982 | 0.0000 | 4970 |
| native_wait_aware_replan | 241 | -2802.632 | 8.5280 | 0.4780 | -9.4840 | 66.0 | -44.39 | 1.0 | 1.0 | -2.00 | 0.982 | 1.0000 | 4970 |
| native_wait_aware_replan | 291 | -11427.618 | 5.4770 | 0.4933 | -6.4636 | 66.0 | -44.43 | 1.0 | 1.0 | -1.84 | 0.982 | 0.0000 | 4971 |
| native_wait_aware_replan | 301 | -3011.818 | 6.1290 | 0.5404 | -7.2098 | 66.0 | -25.37 | 1.0 | 1.0 | -2.00 | 0.982 | 0.5385 | 4971 |
| native_wait_aware_replan | 361 | -3024.825 | 5.6790 | 0.3388 | -6.3566 | 66.0 | -44.47 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 371 | -2623.568 | 8.4550 | 0.3905 | -9.2360 | 66.0 | -44.25 | 1.0 | 1.0 | -2.00 | 0.982 | 0.0000 | 4970 |
| native_wait_aware_replan | 411 | -20610.988 | 16.5030 | 0.4120 | -17.3270 | 66.0 | -44.48 | 1.0 | 1.0 | -0.63 | 0.982 | 0.2308 | 4970 |
| native_wait_aware_replan | 431 | -9106.576 | 5.0400 | 0.4311 | -5.9022 | 66.0 | -44.50 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4972 |
| native_wait_aware_replan | 551 | -2708.144 | 6.5690 | 0.4923 | -7.5536 | 66.0 | -44.03 | 1.0 | 1.0 | -2.00 | 0.982 | 0.9733 | 4970 |
| native_wait_aware_replan | 561 | -9312.926 | 4.8910 | 0.3895 | -5.6700 | 66.0 | -44.47 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| native_wait_aware_replan | 881 | -2684.396 | 4.1510 | 0.5459 | -5.2428 | 66.0 | -44.16 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4971 |
| native_wait_aware_replan | 901 | -17963.226 | 34.9770 | 0.5507 | -36.0784 | 66.0 | -44.09 | 1.0 | 1.0 | -2.00 | 0.982 | 0.2308 | 4970 |
| native_wait_aware_replan | 971 | -9049.347 | 5.9650 | 0.5663 | -7.0976 | 66.0 | -43.66 | 1.0 | 1.0 | -2.00 | 0.982 | 0.0000 | 4971 |
| native_wait_aware_replan | 981 | -11997.002 | 5.4450 | 0.3971 | -6.2392 | 66.0 | -44.17 | 1.0 | 1.0 | -0.60 | 0.982 | 1.0000 | 4970 |
| native_wait_aware_replan | 1221 | -2892.566 | 6.3860 | 0.3663 | -7.1186 | 66.0 | -44.44 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4971 |
| native_wait_aware_replan | 1251 | -11930.187 | 5.1120 | 0.1907 | -5.4934 | 66.0 | -44.37 | 0.0 | 0.0 | 0.00 | 0.000 | 0.3846 | 4971 |
| native_wait_aware_replan | 1401 | -2778.479 | 4.6170 | 0.4471 | -5.5112 | 66.0 | -44.41 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_promotion_replan_vs_interval_ep_reward | not_supported | ep_reward | 16 | -4.0231 | -20.9019 | +15.0145 | 0.56 |
| native_promotion_replan_vs_interval_avg_wait_min | inconclusive | avg_wait_min | 16 | -0.0111 | -0.0270 | +0.0011 | 0.31 |
| native_promotion_replan_vs_interval_score | supported | score | 16 | +0.0765 | +0.0161 | +0.1527 | 0.50 |
| native_promotion_replan_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 16 | +82.9375 | +67.1875 | +98.1891 | 1.00 |
| native_learned_gate_vs_interval_ep_reward | not_supported | ep_reward | 16 | -2.2821 | -12.7448 | +9.3448 | 0.31 |
| native_learned_gate_vs_interval_avg_wait_min | inconclusive | avg_wait_min | 16 | -0.0009 | -0.0231 | +0.0203 | 0.31 |
| native_learned_gate_vs_interval_score | inconclusive | score | 16 | +0.0268 | -0.0182 | +0.0796 | 0.38 |
| native_learned_gate_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 16 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_learned_gate_vs_interval_shared_ppo_gate_replans | supported | shared_ppo_gate_replans | 16 | +2.0000 | +2.0000 | +2.0000 | 1.00 |
| native_learned_gate_vs_interval_ep_reward_noninferiority | supported | ep_reward | 16 | -2.2821 | -12.7448 | +9.3448 | 0.31 |
| native_learned_gate_vs_interval_avg_wait_min_noninferiority | positive_mixed | avg_wait_min | 16 | -0.0009 | -0.0231 | +0.0203 | 0.31 |
| native_wait_aware_replan_vs_interval_ep_reward | not_supported | ep_reward | 16 | -7.3124 | -34.2683 | +11.6571 | 0.19 |
| native_wait_aware_replan_vs_interval_avg_wait_min | inconclusive | avg_wait_min | 16 | -0.0167 | -0.0446 | +0.0014 | 0.38 |
| native_wait_aware_replan_vs_interval_score | supported | score | 16 | +0.0634 | +0.0075 | +0.1335 | 0.38 |
| native_wait_aware_replan_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 16 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_gate_replans | supported | shared_ppo_gate_replans | 16 | +0.5625 | +0.3125 | +0.8125 | 0.56 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_count | supported | shared_ppo_wait_replan_count | 16 | +0.5625 | +0.3125 | +0.8125 | 0.56 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_pressure_mean | supported | shared_ppo_wait_replan_pressure_mean | 16 | +0.2964 | +0.1629 | +0.4361 | 0.56 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_shift_pressure_mean | supported | shared_ppo_wait_replan_shift_pressure_mean | 16 | +0.2076 | +0.1009 | +0.3257 | 0.56 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_gap_ratio_mean | supported | shared_ppo_wait_replan_gap_ratio_mean | 16 | +0.6028 | +0.3383 | +0.8688 | 0.56 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_same_hold_mean | supported | shared_ppo_wait_replan_same_hold_mean | 16 | +0.1519 | +0.0454 | +0.2809 | 0.31 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_same_wait_mean | supported | shared_ppo_wait_replan_same_wait_mean | 16 | +0.5061 | +0.2911 | +0.7333 | 0.56 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_shift_abs_mean_s | supported | shared_ppo_wait_replan_shift_abs_mean_s | 16 | +0.9416 | +0.5193 | +1.3943 | 0.56 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_shift_mean_s | supported | shared_ppo_wait_replan_shift_mean_s | 16 | -0.9416 | -1.3943 | -0.5193 | 0.56 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_actor_base_used_mean | supported | shared_ppo_wait_replan_actor_base_used_mean | 16 | +0.5625 | +0.3125 | +0.8125 | 0.56 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_base_delta_abs_mean_s | supported | shared_ppo_wait_replan_base_delta_abs_mean_s | 16 | +1.1250 | +0.6250 | +1.6250 | 0.56 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_final_delta_abs_mean_s | supported | shared_ppo_wait_replan_final_delta_abs_mean_s | 16 | +1.0210 | +0.5488 | +1.4977 | 0.56 |
| native_wait_aware_replan_vs_interval_upper_plan_target_mean | not_supported | upper_plan_target_mean | 16 | +0.0035 | -0.2421 | +0.2335 | 0.25 |
| native_wait_aware_replan_vs_interval_terminal_launch_shift_mean | not_supported | terminal_launch_shift_mean | 16 | +0.5756 | +0.0000 | +1.7171 | 0.00 |
| native_wait_aware_replan_vs_interval_ep_reward_noninferiority | positive_mixed | ep_reward | 16 | -7.3124 | -34.2683 | +11.6571 | 0.19 |
| native_wait_aware_replan_vs_interval_avg_wait_min_noninferiority | supported | avg_wait_min | 16 | -0.0167 | -0.0446 | +0.0014 | 0.38 |
