# Native Real-Demand Transit Control Validation

merged native simulator passenger loop with public AFC/APC profile mapping

Control profile: `default`.
Demand scale multiplier: `1.0`.

## Sources

| source | rows | series | bins/hour | boundary |
|---|---:|---:|---:|---|
| afc | 1000 | 4 | 1 | AFC station entries, not onboard load or OD |
| apc | 1000 | 4 | 2 | APC route boardings, not onboard occupancy/alighting/OD |

## Paired Checks

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_real_demand_control_score | supported | control_score | 96 | +3496.2558 | +2982.9477 | +3988.3013 | 1.00 |
| native_real_demand_ep_reward | supported | ep_reward | 96 | +3074.8919 | +2560.8985 | +3564.7558 | 0.98 |
| native_real_demand_avg_wait_min | supported | avg_wait_min | 96 | -0.2304 | -0.2327 | -0.2279 | 1.00 |
| native_real_demand_native_avg_board_wait_min | supported | native_avg_board_wait_min | 96 | -0.1760 | -0.1777 | -0.1740 | 1.00 |
| native_real_demand_native_boarded_pax | supported | native_boarded_pax | 96 | +16.7589 | +16.5733 | +16.9227 | 1.00 |
| native_real_demand_native_alighted_pax | supported | native_alighted_pax | 96 | +16.7589 | +16.5733 | +16.9227 | 1.00 |
| native_real_demand_native_completed_throughput_pax | supported | native_completed_throughput_pax | 96 | +16.7589 | +16.5733 | +16.9227 | 1.00 |
| native_real_demand_native_unalighted_pax | not_supported | native_unalighted_pax | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_native_avg_onboard_load | inconclusive | native_avg_onboard_load | 96 | -0.0000 | -0.0000 | +0.0000 | 0.01 |
| native_real_demand_native_service_adjusted | supported | native_service_adjusted | 96 | +1.0000 | +1.0000 | +1.0000 | 1.00 |
| native_real_demand_service_adjustment_signal | supported | service_adjustment_signal | 96 | +0.4190 | +0.4143 | +0.4231 | 1.00 |
| native_real_demand_service_adjustment_wait_reduction_min | supported | service_adjustment_wait_reduction_min | 96 | +0.2304 | +0.2279 | +0.2327 | 1.00 |
| native_real_demand_service_adjustment_board_wait_reduction_min | supported | service_adjustment_board_wait_reduction_min | 96 | +0.1760 | +0.1740 | +0.1777 | 1.00 |
| native_real_demand_service_adjustment_throughput_gain_pax | supported | service_adjustment_throughput_gain_pax | 96 | +16.7589 | +16.5733 | +16.9227 | 1.00 |
| native_real_demand_service_adjustment_lower_lf_drift_reduction | supported | service_adjustment_lower_lf_drift_reduction | 96 | +0.1727 | +0.1706 | +0.1746 | 1.00 |
| native_real_demand_native_raw_avg_wait_min | not_supported | native_raw_avg_wait_min | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_native_raw_native_avg_board_wait_min | not_supported | native_raw_native_avg_board_wait_min | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_native_raw_native_alighted_pax | not_supported | native_raw_native_alighted_pax | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_native_raw_native_completed_throughput_pax | not_supported | native_raw_native_completed_throughput_pax | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_native_raw_LowerLFDrift | not_supported | native_raw_LowerLFDrift | 96 | +0.0000 | -0.0000 | +0.0000 | 0.01 |
| native_real_demand_LowerLFDrift | supported | LowerLFDrift | 96 | -0.1727 | -0.1746 | -0.1706 | 1.00 |
| native_real_demand_UpperHFPower | inconclusive | UpperHFPower | 96 | -0.0000 | -0.0000 | +0.0000 | 0.01 |
| native_real_demand_shared_ppo_wait_replan_count | not_supported | shared_ppo_wait_replan_count | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_shared_ppo_pressure_guard_rejects | not_supported | shared_ppo_pressure_guard_rejects | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_shared_ppo_reward_floor_guard_rejects | not_supported | shared_ppo_reward_floor_guard_rejects | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_shared_ppo_throughput_guard_rejects | not_supported | shared_ppo_throughput_guard_rejects | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_shared_ppo_throughput_floor_project_count | supported | shared_ppo_throughput_floor_project_count | 96 | +0.1042 | +0.0208 | +0.2292 | 0.05 |
| native_real_demand_shared_ppo_throughput_floor_delta_fraction_mean | supported | shared_ppo_throughput_floor_delta_fraction_mean | 96 | -0.0456 | -0.0913 | -0.0097 | 0.05 |
| native_real_demand_shared_ppo_adaptive_drift_guard_rejects | supported | shared_ppo_adaptive_drift_guard_rejects | 96 | +1.9167 | +1.1979 | +2.7917 | 0.31 |
| native_real_demand_shared_ppo_gap_risk_guard_rejects | supported | shared_ppo_gap_risk_guard_rejects | 96 | +0.1458 | +0.0312 | +0.3021 | 0.07 |
| native_real_demand_shared_ppo_target_headway_floor_rejects | supported | shared_ppo_target_headway_floor_rejects | 96 | +0.0938 | +0.0208 | +0.2083 | 0.05 |
| native_real_demand_shared_ppo_target_headway_project_count | not_supported | shared_ppo_target_headway_project_count | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_shared_ppo_wait_replan_adaptive_drift_scale_mean | not_supported | shared_ppo_wait_replan_adaptive_drift_scale_mean | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_shared_ppo_wait_replan_adaptive_drift_hf_to_lf_mean | not_supported | shared_ppo_wait_replan_adaptive_drift_hf_to_lf_mean | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_shared_ppo_wait_replan_throughput_score_mean | not_supported | shared_ppo_wait_replan_throughput_score_mean | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_shared_ppo_wait_replan_throughput_floor_delta_fraction_mean | not_supported | shared_ppo_wait_replan_throughput_floor_delta_fraction_mean | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_shared_ppo_wait_replan_reward_floor_score_mean | not_supported | shared_ppo_wait_replan_reward_floor_score_mean | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_shared_ppo_adaptive_lower_drift_penalty_scale_mean | supported | shared_ppo_adaptive_lower_drift_penalty_scale_mean | 96 | -0.3978 | -0.4040 | -0.3912 | 1.00 |
| native_real_demand_shared_ppo_adaptive_lower_drift_penalty_hf_to_lf_mean | not_supported | shared_ppo_adaptive_lower_drift_penalty_hf_to_lf_mean | 96 | +10.6326 | +9.5603 | +11.8230 | 0.00 |
| native_real_demand_shared_ppo_lower_hf_wait_prior_scale_mean | supported | shared_ppo_lower_hf_wait_prior_scale_mean | 96 | -0.5714 | -0.5782 | -0.5645 | 1.00 |
| native_real_demand_shared_ppo_lower_hf_wait_prior_load_mean | supported | shared_ppo_lower_hf_wait_prior_load_mean | 96 | +0.6362 | +0.6244 | +0.6474 | 1.00 |
| native_real_demand_shared_ppo_lower_hf_wait_prior_queue_mean | supported | shared_ppo_lower_hf_wait_prior_queue_mean | 96 | +0.4249 | +0.3855 | +0.4638 | 1.00 |
| native_real_demand_shared_ppo_lower_hf_wait_prior_schedule_slack_mean | not_supported | shared_ppo_lower_hf_wait_prior_schedule_slack_mean | 96 | -0.0463 | -0.0648 | -0.0283 | 0.30 |
| native_real_demand_shared_ppo_lower_hf_wait_boarding_rescue_mean | supported | shared_ppo_lower_hf_wait_boarding_rescue_mean | 96 | +0.0599 | +0.0530 | +0.0675 | 1.00 |
| native_real_demand_shared_ppo_wait_replan_pressure_override_count | not_supported | shared_ppo_wait_replan_pressure_override_count | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_shared_ppo_wait_replan_pressure_override_mean | not_supported | shared_ppo_wait_replan_pressure_override_mean | 96 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_real_demand_wait_proxy_noninferiority | supported | avg_wait_min | 96 | -0.2304 | -0.2327 | -0.2279 | 1.00 |
| native_real_demand_wait_noninferiority | supported | native_avg_board_wait_min | 96 | -0.1760 | -0.1777 | -0.1740 | 1.00 |
| native_real_demand_alighted_noninferiority | supported | native_alighted_pax | 96 | +16.7589 | +16.5733 | +16.9227 | 1.00 |
| native_real_demand_completed_throughput_noninferiority | supported | native_completed_throughput_pax | 96 | +16.7589 | +16.5733 | +16.9227 | 1.00 |
