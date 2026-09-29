# Native Transit Promotion Replan Validation

This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.
All variants use lower HF wait action prior gain `45.0s` so promotion is validated inside the full Freq-HRL lower-control loop.
Each native batch uses `1` shared-PPO replay update(s).
Runner workers: `16`.

| variant | seed | reward | wait | cv | score | upper decisions | launch shift | gate replans | wait replans | shift | gate | promotion strength | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interval_only | 31 | 21428.321 | 60.4610 | 0.4706 | -61.4022 | 66.0 | +0.06 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| interval_only | 41 | 15997.883 | 85.8140 | 0.3472 | -86.5084 | 66.0 | +0.15 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 51 | 4067.808 | 66.1520 | 0.5372 | -67.2264 | 66.0 | +3.10 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 61 | 20611.980 | 65.3950 | 0.5636 | -66.5222 | 66.0 | +4.98 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| interval_only | 71 | 19972.377 | 63.4070 | 0.5870 | -64.5810 | 66.0 | +5.97 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| interval_only | 81 | 13689.321 | 70.2060 | 0.5706 | -71.3472 | 66.0 | +1.50 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 91 | 2675.006 | 103.5010 | 0.5137 | -104.5284 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.9090 | 4969 |
| interval_only | 101 | 3676.163 | 121.0510 | 0.4306 | -121.9122 | 66.0 | +0.08 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4969 |
| interval_only | 111 | 14028.766 | 65.7980 | 0.5999 | -66.9978 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.8462 | 4970 |
| interval_only | 121 | 14971.402 | 66.1580 | 0.4484 | -67.0548 | 66.0 | +0.48 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4969 |
| interval_only | 131 | 12395.001 | 72.0060 | 0.5848 | -73.1756 | 66.0 | +2.40 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| interval_only | 141 | 9508.877 | 64.9850 | 0.6377 | -66.2604 | 66.0 | +4.81 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 151 | 18261.543 | 62.9030 | 0.5210 | -63.9450 | 66.0 | +3.07 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 161 | 28367.351 | 63.1170 | 0.4465 | -64.0100 | 66.0 | +5.35 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| interval_only | 171 | 5211.415 | 141.2390 | 0.5692 | -142.3774 | 66.0 | +1.34 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| interval_only | 181 | 705.435 | 117.5010 | 0.4862 | -118.4734 | 66.0 | +7.53 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| interval_only | 191 | 22976.436 | 63.1620 | 0.3837 | -63.9294 | 66.0 | +10.72 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0769 | 4969 |
| interval_only | 201 | -496.670 | 139.0840 | 0.5578 | -140.1996 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| interval_only | 211 | 13787.678 | 63.3920 | 0.5656 | -64.5232 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 221 | 15926.294 | 62.3900 | 0.3713 | -63.1326 | 66.0 | +2.04 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 231 | 14696.680 | 63.7720 | 0.5394 | -64.8508 | 66.0 | +2.74 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 241 | 19506.813 | 64.5630 | 0.5475 | -65.6580 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 251 | 19558.966 | 64.1930 | 0.5298 | -65.2526 | 66.0 | +6.93 | 0.0 | 0.0 | 0.00 | 0.000 | 0.4303 | 4970 |
| interval_only | 261 | 26073.194 | 67.3180 | 0.4299 | -68.1778 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 271 | 3297.696 | 119.8290 | 0.6151 | -121.0592 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| interval_only | 281 | 13718.852 | 64.0590 | 0.7090 | -65.4770 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| interval_only | 291 | 17137.130 | 71.2930 | 0.2867 | -71.8664 | 66.0 | +0.06 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| interval_only | 301 | 18936.277 | 62.9660 | 0.4967 | -63.9594 | 66.0 | +15.81 | 0.0 | 0.0 | 0.00 | 0.000 | 0.5385 | 4971 |
| interval_only | 311 | 10426.790 | 63.3750 | 0.6730 | -64.7210 | 66.0 | +7.32 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| interval_only | 321 | 23840.716 | 65.1230 | 0.5282 | -66.1794 | 66.0 | +3.48 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 331 | 22456.766 | 65.1780 | 0.3173 | -65.8126 | 66.0 | +0.78 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| interval_only | 341 | 15037.295 | 77.4600 | 0.3766 | -78.2132 | 66.0 | +0.47 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 31 | 21428.321 | 60.4610 | 0.4706 | -61.4022 | 66.0 | +0.06 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_wait_aware_replan | 41 | 15997.883 | 85.8140 | 0.3472 | -86.5084 | 66.0 | +0.15 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 51 | 4067.808 | 66.1520 | 0.5372 | -67.2264 | 66.0 | +3.10 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 61 | 20611.980 | 65.3950 | 0.5636 | -66.5222 | 66.0 | +4.98 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_wait_aware_replan | 71 | 19972.377 | 63.4070 | 0.5870 | -64.5810 | 66.0 | +5.97 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_wait_aware_replan | 81 | 13689.321 | 70.2060 | 0.5706 | -71.3472 | 66.0 | +1.50 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 91 | 2675.006 | 103.5010 | 0.5137 | -104.5284 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.9090 | 4969 |
| native_wait_aware_replan | 101 | 3676.163 | 121.0510 | 0.4306 | -121.9122 | 66.0 | +0.08 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4969 |
| native_wait_aware_replan | 111 | 14028.766 | 65.7980 | 0.5999 | -66.9978 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.8462 | 4970 |
| native_wait_aware_replan | 121 | 14971.402 | 66.1580 | 0.4484 | -67.0548 | 66.0 | +0.48 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4969 |
| native_wait_aware_replan | 131 | 12395.001 | 72.0060 | 0.5848 | -73.1756 | 66.0 | +2.40 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_wait_aware_replan | 141 | 9508.877 | 64.9850 | 0.6377 | -66.2604 | 66.0 | +4.81 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 151 | 18261.543 | 62.9030 | 0.5210 | -63.9450 | 66.0 | +3.07 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 161 | 28367.351 | 63.1170 | 0.4465 | -64.0100 | 66.0 | +5.35 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_wait_aware_replan | 171 | 5211.415 | 141.2390 | 0.5692 | -142.3774 | 66.0 | +1.34 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| native_wait_aware_replan | 181 | 705.435 | 117.5010 | 0.4862 | -118.4734 | 66.0 | +7.53 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| native_wait_aware_replan | 191 | 22976.436 | 63.1620 | 0.3837 | -63.9294 | 66.0 | +10.72 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0769 | 4969 |
| native_wait_aware_replan | 201 | -496.670 | 139.0840 | 0.5578 | -140.1996 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| native_wait_aware_replan | 211 | 13787.678 | 63.3920 | 0.5656 | -64.5232 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 221 | 15926.294 | 62.3900 | 0.3713 | -63.1326 | 66.0 | +2.04 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 231 | 14696.680 | 63.7720 | 0.5394 | -64.8508 | 66.0 | +2.74 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 241 | 19506.813 | 64.5630 | 0.5475 | -65.6580 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 251 | 19558.966 | 64.1930 | 0.5298 | -65.2526 | 66.0 | +6.93 | 0.0 | 0.0 | 0.00 | 0.000 | 0.4303 | 4970 |
| native_wait_aware_replan | 261 | 26073.194 | 67.3180 | 0.4299 | -68.1778 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 271 | 3297.696 | 119.8290 | 0.6151 | -121.0592 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 1.0000 | 4970 |
| native_wait_aware_replan | 281 | 13718.852 | 64.0590 | 0.7090 | -65.4770 | 66.0 | +0.00 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_wait_aware_replan | 291 | 17137.130 | 71.2930 | 0.2867 | -71.8664 | 66.0 | +0.06 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_wait_aware_replan | 301 | 18936.277 | 62.9660 | 0.4967 | -63.9594 | 66.0 | +15.81 | 0.0 | 0.0 | 0.00 | 0.000 | 0.5385 | 4971 |
| native_wait_aware_replan | 311 | 10426.790 | 63.3750 | 0.6730 | -64.7210 | 66.0 | +7.32 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4971 |
| native_wait_aware_replan | 321 | 23840.716 | 65.1230 | 0.5282 | -66.1794 | 66.0 | +3.48 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 331 | 22456.766 | 65.1780 | 0.3173 | -65.8126 | 66.0 | +0.78 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |
| native_wait_aware_replan | 341 | 15037.295 | 77.4600 | 0.3766 | -78.2132 | 66.0 | +0.47 | 0.0 | 0.0 | 0.00 | 0.000 | 0.0000 | 4970 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_promotion_replan_vs_interval_ep_reward | underpowered | ep_reward | 0 | +nan | +nan | +nan | nan |
| native_promotion_replan_vs_interval_avg_wait_min | underpowered | avg_wait_min | 0 | +nan | +nan | +nan | nan |
| native_promotion_replan_vs_interval_score | underpowered | score | 0 | +nan | +nan | +nan | nan |
| native_promotion_replan_vs_interval_upper_plan_decisions | underpowered | upper_plan_decisions | 0 | +nan | +nan | +nan | nan |
| native_wait_aware_replan_vs_interval_ep_reward | not_supported | ep_reward | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_avg_wait_min | not_supported | avg_wait_min | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_score | not_supported | score | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_gate_replans | not_supported | shared_ppo_gate_replans | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_target_headway_guard_rejects | not_supported | shared_ppo_target_headway_guard_rejects | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_target_headway_project_count | not_supported | shared_ppo_target_headway_project_count | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_target_headway_project_correction_mean_s | not_supported | shared_ppo_target_headway_project_correction_mean_s | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_pressure_guard_rejects | not_supported | shared_ppo_pressure_guard_rejects | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_soft_pressure_cap_count | supported | shared_ppo_soft_pressure_cap_count | 32 | +0.1875 | +0.0312 | +0.3750 | 0.12 |
| native_wait_aware_replan_vs_interval_shared_ppo_soft_pressure_cap_scale_mean | supported | shared_ppo_soft_pressure_cap_scale_mean | 32 | -0.0049 | -0.0106 | -0.0001 | 0.12 |
| native_wait_aware_replan_vs_interval_shared_ppo_reward_floor_guard_rejects | not_supported | shared_ppo_reward_floor_guard_rejects | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_confirm_guard_rejects | not_supported | shared_ppo_confirm_guard_rejects | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_value_guard_rejects | supported | shared_ppo_value_guard_rejects | 32 | +149.9375 | +144.3117 | +155.1562 | 1.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_throughput_guard_rejects | not_supported | shared_ppo_throughput_guard_rejects | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_throughput_floor_project_count | inconclusive | shared_ppo_throughput_floor_project_count | 32 | +0.1562 | +0.0000 | +0.4062 | 0.06 |
| native_wait_aware_replan_vs_interval_shared_ppo_throughput_floor_delta_fraction_mean | inconclusive | shared_ppo_throughput_floor_delta_fraction_mean | 32 | -0.0596 | -0.1492 | +0.0000 | 0.06 |
| native_wait_aware_replan_vs_interval_shared_ppo_adaptive_drift_guard_rejects | supported | shared_ppo_adaptive_drift_guard_rejects | 32 | +1.6562 | +0.4688 | +3.1562 | 0.25 |
| native_wait_aware_replan_vs_interval_shared_ppo_gap_risk_guard_rejects | inconclusive | shared_ppo_gap_risk_guard_rejects | 32 | +0.0312 | +0.0000 | +0.0938 | 0.03 |
| native_wait_aware_replan_vs_interval_shared_ppo_active_target_headway_floor_rejects | not_supported | shared_ppo_active_target_headway_floor_rejects | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_target_headway_floor_rejects | supported | shared_ppo_target_headway_floor_rejects | 32 | +0.2188 | +0.0312 | +0.4375 | 0.12 |
| native_wait_aware_replan_vs_interval_shared_ppo_base_delta_guard_rejects | not_supported | shared_ppo_base_delta_guard_rejects | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_final_delta_guard_rejects | not_supported | shared_ppo_final_delta_guard_rejects | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_final_delta_floor_rejects | inconclusive | shared_ppo_final_delta_floor_rejects | 32 | +0.1562 | +0.0000 | +0.3750 | 0.09 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_count | not_supported | shared_ppo_wait_replan_count | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_pressure_mean | not_supported | shared_ppo_wait_replan_pressure_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_shift_pressure_mean | not_supported | shared_ppo_wait_replan_shift_pressure_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_gap_ratio_mean | not_supported | shared_ppo_wait_replan_gap_ratio_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_gap_risk_scale_mean | not_supported | shared_ppo_wait_replan_gap_risk_scale_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_adaptive_drift_scale_mean | not_supported | shared_ppo_wait_replan_adaptive_drift_scale_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_adaptive_drift_hf_to_lf_mean | not_supported | shared_ppo_wait_replan_adaptive_drift_hf_to_lf_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_throughput_score_mean | not_supported | shared_ppo_wait_replan_throughput_score_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_throughput_floor_delta_fraction_mean | not_supported | shared_ppo_wait_replan_throughput_floor_delta_fraction_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_reward_floor_score_mean | not_supported | shared_ppo_wait_replan_reward_floor_score_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_value_guard_score_mean | not_supported | shared_ppo_wait_replan_value_guard_score_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_value_guard_scale_mean | not_supported | shared_ppo_wait_replan_value_guard_scale_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_value_guard_candidate_count_mean | not_supported | shared_ppo_wait_replan_value_guard_candidate_count_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_adaptive_lower_drift_penalty_scale_mean | not_supported | shared_ppo_adaptive_lower_drift_penalty_scale_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_adaptive_lower_drift_penalty_hf_to_lf_mean | not_supported | shared_ppo_adaptive_lower_drift_penalty_hf_to_lf_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_lower_hf_wait_prior_scale_mean | not_supported | shared_ppo_lower_hf_wait_prior_scale_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_lower_hf_wait_prior_load_mean | not_supported | shared_ppo_lower_hf_wait_prior_load_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_lower_hf_wait_prior_queue_mean | not_supported | shared_ppo_lower_hf_wait_prior_queue_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_lower_hf_wait_prior_schedule_slack_mean | not_supported | shared_ppo_lower_hf_wait_prior_schedule_slack_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_pressure_override_count | not_supported | shared_ppo_wait_replan_pressure_override_count | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_pressure_override_mean | not_supported | shared_ppo_wait_replan_pressure_override_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_same_hold_mean | not_supported | shared_ppo_wait_replan_same_hold_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_same_wait_mean | not_supported | shared_ppo_wait_replan_same_wait_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_shift_abs_mean_s | not_supported | shared_ppo_wait_replan_shift_abs_mean_s | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_shift_mean_s | not_supported | shared_ppo_wait_replan_shift_mean_s | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_actor_base_used_mean | not_supported | shared_ppo_wait_replan_actor_base_used_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_base_delta_abs_mean_s | not_supported | shared_ppo_wait_replan_base_delta_abs_mean_s | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_shared_ppo_wait_replan_final_delta_abs_mean_s | not_supported | shared_ppo_wait_replan_final_delta_abs_mean_s | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_freq_wait_lower_improvement_credit_mean | not_supported | freq_wait_lower_improvement_credit_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_freq_wait_lower_net_mean | not_supported | freq_wait_lower_net_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_upper_plan_target_mean | not_supported | upper_plan_target_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_terminal_launch_shift_mean | not_supported | terminal_launch_shift_mean | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_ep_reward_noninferiority | supported | ep_reward | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_vs_interval_avg_wait_min_noninferiority | supported | avg_wait_min | 32 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_promotion_strength_ge1_vs_interval_ep_reward | underpowered | ep_reward | 5 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_promotion_strength_ge1_vs_interval_avg_wait_min | underpowered | avg_wait_min | 5 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_promotion_strength_ge1_vs_interval_score | underpowered | score | 5 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_promotion_strength_ge1_vs_interval_shared_ppo_wait_replan_count | underpowered | shared_ppo_wait_replan_count | 5 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_promotion_strength_ge1_vs_interval_shared_ppo_wait_replan_pressure_override_count | underpowered | shared_ppo_wait_replan_pressure_override_count | 5 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_headway_cv_ge050_vs_interval_ep_reward | underpowered | ep_reward | 19 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_headway_cv_ge050_vs_interval_avg_wait_min | underpowered | avg_wait_min | 19 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_headway_cv_ge050_vs_interval_score | underpowered | score | 19 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_headway_cv_ge050_vs_interval_shared_ppo_wait_replan_count | underpowered | shared_ppo_wait_replan_count | 19 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_headway_cv_ge050_vs_interval_shared_ppo_wait_replan_pressure_override_count | underpowered | shared_ppo_wait_replan_pressure_override_count | 19 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_upper_plan_target_ge350_vs_interval_ep_reward | underpowered | ep_reward | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_upper_plan_target_ge350_vs_interval_avg_wait_min | underpowered | avg_wait_min | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_upper_plan_target_ge350_vs_interval_score | underpowered | score | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_upper_plan_target_ge350_vs_interval_shared_ppo_wait_replan_count | underpowered | shared_ppo_wait_replan_count | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_wait_aware_replan_control_upper_plan_target_ge350_vs_interval_shared_ppo_wait_replan_pressure_override_count | underpowered | shared_ppo_wait_replan_pressure_override_count | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
