# Native Transit Promotion Replan Validation

This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.
All variants use lower HF wait action prior gain `45.0s` so promotion is validated inside the full Freq-HRL lower-control loop.

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
| native_learned_gate | 31 | -7212.534 | 3.7970 | 0.4776 | -4.7522 | 73.0 | 26.0 | 0.982 | 0.3846 | 4971 |
| native_learned_gate | 41 | -11675.006 | 5.5340 | 0.3188 | -6.1716 | 68.0 | 6.0 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 51 | -20906.998 | 35.5300 | 0.6336 | -36.7972 | 72.0 | 21.0 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 61 | -3341.828 | 5.0690 | 0.6153 | -6.2996 | 70.0 | 15.0 | 0.982 | 0.0000 | 4970 |
| native_learned_gate | 71 | -3145.498 | 4.0920 | 0.5597 | -5.2114 | 68.0 | 8.0 | 0.982 | 0.0000 | 4971 |
| native_learned_gate | 81 | -12299.443 | 4.7310 | 0.5713 | -5.8736 | 72.0 | 21.0 | 0.982 | 0.0769 | 4970 |
| native_learned_gate | 91 | -18278.120 | 4.6420 | 0.6120 | -5.8660 | 70.0 | 16.0 | 0.982 | 0.0000 | 4969 |
| native_learned_gate | 101 | -20036.709 | 19.9890 | 0.5610 | -21.1110 | 74.0 | 31.0 | 0.982 | 1.0000 | 4971 |
| native_learned_gate | 111 | -14011.553 | 4.1030 | 0.6102 | -5.3234 | 68.0 | 7.0 | 0.982 | 1.0000 | 4970 |
| native_learned_gate | 121 | -12412.966 | 6.1700 | 0.2303 | -6.6306 | 70.0 | 12.0 | 0.982 | 0.3573 | 4970 |
| native_learned_gate | 131 | -18582.156 | 6.0050 | 0.6346 | -7.2742 | 71.0 | 21.0 | 0.982 | 0.5559 | 4971 |
| native_learned_gate | 141 | -17274.421 | 38.3660 | 0.5727 | -39.5114 | 68.0 | 4.0 | 0.982 | 0.0000 | 4970 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_promotion_replan_vs_interval_ep_reward | inconclusive | ep_reward | 12 | +0.0168 | -11.2749 | +16.2435 | 0.25 |
| native_promotion_replan_vs_interval_avg_wait_min | not_supported | avg_wait_min | 12 | +0.0092 | -0.0373 | +0.0716 | 0.50 |
| native_promotion_replan_vs_interval_score | not_supported | score | 12 | -0.0436 | -0.1878 | +0.0749 | 0.50 |
| native_promotion_replan_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 12 | +49.4167 | +36.4146 | +62.0000 | 1.00 |
| native_learned_gate_vs_interval_ep_reward | positive_mixed | ep_reward | 12 | +11.1923 | -9.1302 | +33.6636 | 0.58 |
| native_learned_gate_vs_interval_avg_wait_min | inconclusive | avg_wait_min | 12 | -0.0066 | -0.0523 | +0.0438 | 0.33 |
| native_learned_gate_vs_interval_score | supported | score | 12 | +0.0385 | +0.0047 | +0.0752 | 0.58 |
| native_learned_gate_vs_interval_upper_plan_decisions | supported | upper_plan_decisions | 12 | +4.3333 | +3.2500 | +5.5000 | 1.00 |
| native_learned_gate_vs_interval_shared_ppo_gate_replans | supported | shared_ppo_gate_replans | 12 | +15.6667 | +11.0000 | +20.4167 | 1.00 |
