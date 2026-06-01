# Native Transit Promotion Replan Validation

This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.

| variant | seed | reward | wait | cv | score | upper decisions | gate replans | gate | promotion strength | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interval_only | 31 | -18839.663 | 4.3670 | 0.4202 | -5.2074 | 66.0 | 0.0 | 0.000 | 1.0000 | 4970 |
| interval_only | 41 | -15570.082 | 5.0160 | 0.5471 | -6.1102 | 66.0 | 0.0 | 0.000 | 0.8462 | 4970 |
| interval_only | 51 | -6901.075 | 5.6880 | 0.5385 | -6.7650 | 66.0 | 0.0 | 0.000 | 0.5385 | 4970 |
| interval_only | 61 | -18470.286 | 6.9880 | 0.4890 | -7.9660 | 66.0 | 0.0 | 0.000 | 0.8462 | 4971 |
| interval_only | 71 | -2868.419 | 5.1260 | 0.4610 | -6.0480 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| interval_only | 81 | -9074.652 | 3.9960 | 0.5666 | -5.1292 | 66.0 | 0.0 | 0.000 | 1.0000 | 4970 |
| interval_only | 91 | -2758.308 | 4.2210 | 0.3639 | -4.9488 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| interval_only | 101 | -15815.869 | 5.2060 | 0.5632 | -6.3324 | 66.0 | 0.0 | 0.000 | 1.0000 | 4969 |
| native_promotion_replan | 31 | -7537.299 | 5.3680 | 0.6935 | -6.7550 | 116.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 41 | -3013.692 | 3.9690 | 0.4081 | -4.7852 | 104.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 51 | -3815.505 | 5.3340 | 0.5111 | -6.3562 | 128.0 | 0.0 | 0.000 | 0.0769 | 4970 |
| native_promotion_replan | 61 | -6313.737 | 4.6570 | 0.5714 | -5.7998 | 130.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 71 | -17127.753 | 37.8430 | 0.6814 | -39.2058 | 106.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 81 | -19799.748 | 25.8290 | 0.7592 | -27.3474 | 124.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 91 | -18842.940 | 3.9120 | 0.6547 | -5.2214 | 151.0 | 0.0 | 0.000 | 0.3071 | 4970 |
| native_promotion_replan | 101 | -20437.032 | 48.4480 | 0.2936 | -49.0352 | 135.0 | 0.0 | 0.000 | 0.3846 | 4971 |
| native_learned_gate | 31 | -11152.623 | 22.4920 | 0.3066 | -23.1052 | 125.0 | 87.0 | 0.972 | 1.0000 | 4971 |
| native_learned_gate | 41 | -5027.481 | 4.3800 | 0.6075 | -5.5950 | 103.0 | 57.0 | 0.977 | 0.0000 | 4970 |
| native_learned_gate | 51 | -19129.171 | 8.5570 | 0.5116 | -9.5802 | 156.0 | 136.0 | 0.977 | 0.0000 | 4970 |
| native_learned_gate | 61 | -5502.469 | 4.6990 | 0.5957 | -5.8904 | 119.0 | 81.0 | 0.974 | 0.8462 | 4971 |
| native_learned_gate | 71 | -19876.736 | 10.1390 | 0.4388 | -11.0166 | 85.0 | 31.0 | 0.971 | 1.0000 | 4970 |
| native_learned_gate | 81 | -4253.486 | 3.9910 | 0.5148 | -5.0206 | 100.0 | 52.0 | 0.973 | 1.0000 | 4970 |
| native_learned_gate | 91 | -6269.659 | 9.4210 | 0.6109 | -10.6428 | 115.0 | 73.0 | 0.976 | 0.5385 | 4970 |
| native_learned_gate | 101 | -2691.502 | 4.6050 | 0.3987 | -5.4024 | 166.0 | 141.0 | 0.978 | 0.0000 | 4970 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_promotion_replan_vs_interval_ep_reward | not_supported | ep_reward | 8 | -823.6690 | -8513.0658 | +6865.2066 | 0.50 |
| native_promotion_replan_vs_interval_avg_wait_min | not_supported | avg_wait_min | 8 | +11.8440 | +1.5025 | +23.8893 | 0.50 |
| native_promotion_replan_vs_interval_score | not_supported | score | 8 | -11.9999 | -23.7297 | -1.6135 | 0.38 |
| native_promotion_replan_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 8 | +58.2500 | +48.7500 | +68.3750 | 1.00 |
| native_learned_gate_vs_interval_ep_reward | positive_mixed | ep_reward | 8 | +2049.4034 | -6341.6198 | +9232.5186 | 0.62 |
| native_learned_gate_vs_interval_avg_wait_min | not_supported | avg_wait_min | 8 | +3.4595 | -0.0753 | +8.3288 | 0.50 |
| native_learned_gate_vs_interval_score | not_supported | score | 8 | -3.4683 | -8.2290 | +0.0679 | 0.50 |
| native_learned_gate_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 8 | +55.1250 | +38.4969 | +73.1250 | 1.00 |
| native_learned_gate_vs_interval_shared_ppo_gate_replans | supported | shared_ppo_gate_replans | 8 | +82.2500 | +58.8719 | +107.6312 | 1.00 |
