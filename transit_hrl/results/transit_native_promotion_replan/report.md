# Native Transit Promotion Replan Validation

This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.

| variant | seed | reward | wait | cv | score | upper decisions | gate replans | gate | promotion strength | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interval_only | 31 | -7278.462 | 3.7860 | 0.4767 | -4.7394 | 66.0 | 0.0 | 0.000 | 0.3846 | 4971 |
| interval_only | 41 | -11894.049 | 5.2820 | 0.4405 | -6.1630 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| interval_only | 51 | -20939.222 | 35.7410 | 0.6129 | -36.9668 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| interval_only | 61 | -3348.018 | 5.0690 | 0.6153 | -6.2996 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| interval_only | 71 | -3138.154 | 4.0940 | 0.5596 | -5.2132 | 66.0 | 0.0 | 0.000 | 0.0000 | 4971 |
| interval_only | 81 | -12307.562 | 4.7390 | 0.4932 | -5.7254 | 66.0 | 0.0 | 0.000 | 0.0769 | 4970 |
| interval_only | 91 | -18284.120 | 4.6390 | 0.6119 | -5.8628 | 66.0 | 0.0 | 0.000 | 0.0000 | 4969 |
| interval_only | 101 | -19975.330 | 19.9730 | 0.6149 | -21.2028 | 66.0 | 0.0 | 0.000 | 1.0000 | 4971 |
| native_promotion_replan | 31 | -7291.724 | 3.7950 | 0.5799 | -4.9548 | 140.0 | 0.0 | 0.000 | 0.3846 | 4971 |
| native_promotion_replan | 41 | -11726.429 | 5.3190 | 0.5307 | -6.3804 | 85.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 51 | -20982.387 | 36.0870 | 0.7650 | -37.6170 | 130.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 61 | -3352.145 | 5.0690 | 0.6153 | -6.2996 | 112.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 71 | -3140.083 | 4.0970 | 0.5732 | -5.2434 | 114.0 | 0.0 | 0.000 | 0.0000 | 4971 |
| native_promotion_replan | 81 | -12349.898 | 4.7430 | 0.5805 | -5.9040 | 131.0 | 0.0 | 0.000 | 0.0769 | 4970 |
| native_promotion_replan | 91 | -18294.634 | 4.6390 | 0.6117 | -5.8624 | 109.0 | 0.0 | 0.000 | 0.0000 | 4969 |
| native_promotion_replan | 101 | -19966.345 | 19.9670 | 0.5565 | -21.0800 | 154.0 | 0.0 | 0.000 | 1.0000 | 4971 |
| native_learned_gate | 31 | -7257.435 | 3.8140 | 0.4770 | -4.7680 | 73.0 | 26.0 | 0.982 | 0.3846 | 4971 |
| native_learned_gate | 41 | -11623.902 | 5.2490 | 0.4874 | -6.2238 | 68.0 | 6.0 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 51 | -20919.147 | 35.5510 | 0.6321 | -36.8152 | 72.0 | 21.0 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 61 | -3341.828 | 5.0690 | 0.6153 | -6.2996 | 70.0 | 15.0 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 71 | -3145.498 | 4.0920 | 0.5597 | -5.2114 | 68.0 | 8.0 | 0.982 | 0.0000 | 4971 |
| native_learned_gate | 81 | -12285.649 | 4.7340 | 0.5332 | -5.8004 | 72.0 | 21.0 | 0.982 | 0.0769 | 4970 |
| native_learned_gate | 91 | -18284.647 | 4.6390 | 0.6119 | -5.8628 | 70.0 | 16.0 | 0.982 | 0.0000 | 4969 |
| native_learned_gate | 101 | -20039.607 | 19.9900 | 0.6037 | -21.1974 | 74.0 | 31.0 | 0.982 | 1.0000 | 4971 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_promotion_replan_vs_interval_ep_reward | inconclusive | ep_reward | 8 | +7.6590 | -25.3803 | +56.1720 | 0.25 |
| native_promotion_replan_vs_interval_avg_wait_min | not_supported | avg_wait_min | 8 | +0.0491 | +0.0004 | +0.1365 | 0.12 |
| native_promotion_replan_vs_interval_score | not_supported | score | 8 | -0.1461 | -0.3194 | -0.0107 | 0.25 |
| native_promotion_replan_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 8 | +55.8750 | +42.8719 | +69.3781 | 1.00 |
| native_learned_gate_vs_interval_ep_reward | positive_mixed | ep_reward | 8 | +33.4005 | -15.6374 | +105.4996 | 0.62 |
| native_learned_gate_vs_interval_avg_wait_min | positive_mixed | avg_wait_min | 8 | -0.0231 | -0.0759 | +0.0095 | 0.50 |
| native_learned_gate_vs_interval_score | not_supported | score | 8 | -0.0007 | -0.0384 | +0.0461 | 0.38 |
| native_learned_gate_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 8 | +4.8750 | +3.5000 | +6.2500 | 1.00 |
| native_learned_gate_vs_interval_shared_ppo_gate_replans | supported | shared_ppo_gate_replans | 8 | +18.0000 | +12.7500 | +23.3750 | 1.00 |
