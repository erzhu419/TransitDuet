# Native Transit Promotion Replan Validation

This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.
All variants use lower HF wait action prior gain `45.0s` so promotion is validated inside the full Freq-HRL lower-control loop.
Each native batch uses `3` shared-PPO replay update(s).
Runner workers: `8`.

| variant | seed | reward | wait | cv | score | upper decisions | launch shift | gate replans | wait replans | shift | gate | promotion strength | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interval_only | 151 | -5259.386 | 10.7920 | 0.4269 | -11.6458 | 66.0 | -44.40 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 161 | -2776.914 | 4.3580 | 0.3749 | -5.1078 | 66.0 | -44.23 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| interval_only | 171 | -16021.407 | 20.5610 | 0.3591 | -21.2792 | 66.0 | -44.50 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| interval_only | 181 | -19603.454 | 14.5230 | 0.7172 | -15.9574 | 66.0 | -42.94 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_promotion_replan | 151 | -5138.408 | 10.5870 | 0.4871 | -11.5612 | 127.0 | -44.40 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 161 | -2775.169 | 4.3590 | 0.3749 | -5.1088 | 104.0 | -44.29 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_promotion_replan | 171 | -16068.466 | 20.6930 | 0.4067 | -21.5064 | 86.0 | -44.50 | 0.0 | 0.0 | 0.00 | 0.000 | 0.8462 | 4970 |
| native_promotion_replan | 181 | -19601.310 | 14.5230 | 0.7172 | -15.9574 | 118.0 | -41.50 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_learned_gate | 151 | -5297.088 | 10.7670 | 0.4858 | -11.7386 | 66.0 | -44.40 | 2.0 | 0.0 | 0.00 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 161 | -2778.141 | 4.3590 | 0.3749 | -5.1088 | 66.0 | -44.29 | 2.0 | 0.0 | 0.00 | 0.982 | 0.0000 | 4971 |
| native_learned_gate | 171 | -16021.407 | 20.5610 | 0.3591 | -21.2792 | 66.0 | -44.50 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| native_learned_gate | 181 | -19607.067 | 14.5230 | 0.7172 | -15.9574 | 66.0 | -43.00 | 2.0 | 0.0 | 0.00 | 0.982 | 0.0000 | 4971 |
| native_wait_aware_replan | 151 | -3818.194 | 7.9620 | 0.4839 | -8.9298 | 66.0 | -44.34 | 2.0 | 2.0 | -17.74 | 0.982 | 0.5385 | 4970 |
| native_wait_aware_replan | 161 | -4146.559 | 4.6860 | 0.4469 | -5.5798 | 66.0 | -44.15 | 2.0 | 2.0 | -13.84 | 0.982 | 0.0000 | 4971 |
| native_wait_aware_replan | 171 | -2641.077 | 9.8040 | 0.4948 | -10.7936 | 66.0 | -44.36 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 181 | -18627.706 | 15.7560 | 0.5194 | -16.7948 | 66.0 | -44.44 | 2.0 | 2.0 | -15.66 | 0.982 | 0.0769 | 4971 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_promotion_replan_vs_interval_ep_reward | positive_mixed | ep_reward | 4 | +19.4520 | -34.7583 | +91.1698 | 0.75 |
| native_promotion_replan_vs_interval_avg_wait_min | inconclusive | avg_wait_min | 4 | -0.0180 | -0.1535 | +0.0990 | 0.25 |
| native_promotion_replan_vs_interval_score | not_supported | score | 4 | -0.0359 | -0.1704 | +0.0632 | 0.25 |
| native_promotion_replan_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 4 | +42.7500 | +28.0000 | +56.5000 | 1.00 |
| native_learned_gate_vs_interval_ep_reward | not_supported | ep_reward | 4 | -10.6355 | -28.5832 | -0.6135 | 0.00 |
| native_learned_gate_vs_interval_avg_wait_min | inconclusive | avg_wait_min | 4 | -0.0060 | -0.0188 | +0.0008 | 0.25 |
| native_learned_gate_vs_interval_score | not_supported | score | 4 | -0.0235 | -0.0696 | +0.0000 | 0.00 |
| native_learned_gate_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 4 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_learned_gate_vs_interval_shared_ppo_gate_replans | supported | shared_ppo_gate_replans | 4 | +1.5000 | +0.5000 | +2.0000 | 0.75 |
| native_learned_gate_vs_interval_ep_reward_noninferiority | positive_mixed | ep_reward | 4 | -10.6355 | -28.5832 | -0.6135 | 0.00 |
| native_learned_gate_vs_interval_avg_wait_min_noninferiority | supported | avg_wait_min | 4 | -0.0060 | -0.0188 | +0.0008 | 0.25 |
| native_wait_aware_replan_vs_interval_ep_reward | positive_mixed | ep_reward | 4 | +3606.9063 | -666.9357 | +10279.1845 | 0.75 |
| native_wait_aware_replan_vs_interval_avg_wait_min | positive_mixed | avg_wait_min | 4 | -3.0065 | -7.9857 | +0.7805 | 0.50 |
| native_wait_aware_replan_vs_interval_score | positive_mixed | score | 4 | +2.9730 | -0.6547 | +7.7462 | 0.50 |
| native_wait_aware_replan_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 4 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_gate_replans | supported | shared_ppo_gate_replans | 4 | +1.5000 | +0.5000 | +2.0000 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_count | supported | shared_ppo_wait_replan_count | 4 | +1.5000 | +0.5000 | +2.0000 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_shift_abs_mean_s | supported | shared_ppo_wait_replan_shift_abs_mean_s | 4 | +11.8103 | +3.9159 | +16.7620 | 0.75 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_shift_mean_s | supported | shared_ppo_wait_replan_shift_mean_s | 4 | -11.8103 | -16.7620 | -3.9159 | 0.75 |
| native_wait_aware_replan_vs_interval_upper_plan_target_mean | supported | upper_plan_target_mean | 4 | -2.8813 | -3.5169 | -2.3232 | 1.00 |
| native_wait_aware_replan_vs_interval_terminal_launch_shift_mean | inconclusive | terminal_launch_shift_mean | 4 | -0.3044 | -1.1107 | +0.1250 | 0.25 |
| native_wait_aware_replan_vs_interval_ep_reward_noninferiority | positive_mixed | ep_reward | 4 | +3606.9063 | -666.9357 | +10279.1845 | 0.75 |
| native_wait_aware_replan_vs_interval_avg_wait_min_noninferiority | positive_mixed | avg_wait_min | 4 | -3.0065 | -7.9857 | +0.7805 | 0.50 |
