# Native Transit Promotion Replan Validation

This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.
All variants use lower HF wait action prior gain `45.0s` so promotion is validated inside the full Freq-HRL lower-control loop.
Each native batch uses `3` shared-PPO replay update(s).
Runner workers: `1`.

| variant | seed | reward | wait | cv | score | upper decisions | gate replans | gate | promotion strength | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interval_only | 31 | -7230.725 | 3.7960 | 0.5274 | -4.8508 | 66.0 | 0.0 | 0.000 | 0.3846 | 4971 |
| interval_only | 41 | -11777.277 | 5.3370 | 0.4010 | -6.1390 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| interval_only | 51 | -20933.385 | 35.7370 | 0.6129 | -36.9628 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| interval_only | 61 | -3348.018 | 5.0690 | 0.6153 | -6.2996 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| interval_only | 71 | -3138.154 | 4.0940 | 0.5596 | -5.2132 | 66.0 | 0.0 | 0.000 | 0.0000 | 4971 |
| interval_only | 81 | -12353.228 | 4.7430 | 0.5804 | -5.9038 | 66.0 | 0.0 | 0.000 | 0.0769 | 4970 |
| interval_only | 91 | -18277.497 | 4.6420 | 0.6120 | -5.8660 | 66.0 | 0.0 | 0.000 | 0.0000 | 4969 |
| interval_only | 101 | -19976.740 | 19.9780 | 0.6352 | -21.2484 | 66.0 | 0.0 | 0.000 | 1.0000 | 4971 |
| interval_only | 111 | -14021.049 | 4.1030 | 0.6109 | -5.3248 | 66.0 | 0.0 | 0.000 | 1.0000 | 4970 |
| interval_only | 121 | -12398.018 | 6.1630 | 0.2259 | -6.6148 | 66.0 | 0.0 | 0.000 | 0.3573 | 4970 |
| interval_only | 131 | -18589.866 | 6.0050 | 0.6346 | -7.2742 | 66.0 | 0.0 | 0.000 | 0.5559 | 4971 |
| interval_only | 141 | -17267.582 | 38.4400 | 0.5734 | -39.5868 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 31 | -7232.572 | 3.7750 | 0.5277 | -4.8304 | 140.0 | 0.0 | 0.000 | 0.3846 | 4971 |
| native_promotion_replan | 41 | -11787.024 | 5.3680 | 0.6483 | -6.6646 | 85.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 51 | -20971.967 | 36.0360 | 0.7245 | -37.4850 | 130.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 61 | -3352.145 | 5.0690 | 0.6153 | -6.2996 | 112.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_promotion_replan | 71 | -3140.083 | 4.0970 | 0.5732 | -5.2434 | 114.0 | 0.0 | 0.000 | 0.0000 | 4971 |
| native_promotion_replan | 81 | -12359.770 | 4.7380 | 0.5171 | -5.7722 | 131.0 | 0.0 | 0.000 | 0.0769 | 4970 |
| native_promotion_replan | 91 | -18286.736 | 4.6420 | 0.6118 | -5.8656 | 109.0 | 0.0 | 0.000 | 0.0000 | 4969 |
| native_promotion_replan | 101 | -19968.947 | 19.9720 | 0.5481 | -21.0682 | 154.0 | 0.0 | 0.000 | 1.0000 | 4971 |
| native_promotion_replan | 111 | -14014.961 | 4.0920 | 0.6505 | -5.3930 | 90.0 | 0.0 | 0.000 | 1.0000 | 4970 |
| native_promotion_replan | 121 | -12324.284 | 6.1430 | 0.2267 | -6.5964 | 109.0 | 0.0 | 0.000 | 0.3573 | 4970 |
| native_promotion_replan | 131 | -18594.268 | 6.0050 | 0.6346 | -7.2742 | 134.0 | 0.0 | 0.000 | 0.5559 | 4971 |
| native_promotion_replan | 141 | -17278.581 | 38.2800 | 0.5171 | -39.3142 | 77.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| native_learned_gate | 31 | -7202.202 | 3.8060 | 0.5274 | -4.8608 | 66.0 | 2.0 | 0.982 | 0.3846 | 4971 |
| native_learned_gate | 41 | -11796.721 | 5.3540 | 0.5823 | -6.5186 | 66.0 | 1.0 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 51 | -20898.101 | 35.5180 | 0.6349 | -36.7878 | 66.0 | 2.0 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 61 | -3341.611 | 5.0700 | 0.6153 | -6.3006 | 66.0 | 2.0 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 71 | -3138.154 | 4.0940 | 0.5596 | -5.2132 | 66.0 | 0.0 | 0.000 | 0.0000 | 4971 |
| native_learned_gate | 81 | -12298.290 | 4.7380 | 0.5681 | -5.8742 | 66.0 | 2.0 | 0.982 | 0.0769 | 4970 |
| native_learned_gate | 91 | -18275.403 | 4.6420 | 0.6118 | -5.8656 | 66.0 | 2.0 | 0.982 | 0.0000 | 4969 |
| native_learned_gate | 101 | -20032.226 | 19.9890 | 0.5610 | -21.1110 | 66.0 | 2.0 | 0.982 | 1.0000 | 4971 |
| native_learned_gate | 111 | -14026.279 | 4.1050 | 0.6221 | -5.3492 | 66.0 | 1.0 | 0.982 | 1.0000 | 4970 |
| native_learned_gate | 121 | -12399.257 | 6.1630 | 0.2303 | -6.6236 | 66.0 | 2.0 | 0.982 | 0.3573 | 4970 |
| native_learned_gate | 131 | -18591.391 | 6.0050 | 0.6346 | -7.2742 | 66.0 | 2.0 | 0.982 | 0.5559 | 4971 |
| native_learned_gate | 141 | -17267.582 | 38.4400 | 0.5734 | -39.5868 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_promotion_replan_vs_interval_ep_reward | inconclusive | ep_reward | 12 | +0.0168 | -11.2749 | +16.2435 | 0.25 |
| native_promotion_replan_vs_interval_avg_wait_min | not_supported | avg_wait_min | 12 | +0.0092 | -0.0373 | +0.0716 | 0.50 |
| native_promotion_replan_vs_interval_score | not_supported | score | 12 | -0.0436 | -0.1878 | +0.0749 | 0.50 |
| native_promotion_replan_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 12 | +49.4167 | +36.4146 | +62.0000 | 1.00 |
| native_learned_gate_vs_interval_ep_reward | inconclusive | ep_reward | 12 | +3.6935 | -12.3283 | +17.8474 | 0.42 |
| native_learned_gate_vs_interval_avg_wait_min | inconclusive | avg_wait_min | 12 | -0.0153 | -0.0530 | +0.0060 | 0.17 |
| native_learned_gate_vs_interval_score | not_supported | score | 12 | -0.0068 | -0.0904 | +0.0555 | 0.33 |
| native_learned_gate_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 12 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| native_learned_gate_vs_interval_shared_ppo_gate_replans | supported | shared_ppo_gate_replans | 12 | +1.5000 | +1.0813 | +1.9167 | 0.83 |
| native_learned_gate_vs_interval_ep_reward_noninferiority | supported | ep_reward | 12 | +3.6935 | -12.3283 | +17.8474 | 0.42 |
| native_learned_gate_vs_interval_avg_wait_min_noninferiority | supported | avg_wait_min | 12 | -0.0153 | -0.0530 | +0.0060 | 0.17 |
