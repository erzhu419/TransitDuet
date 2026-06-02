# Native Transit Promotion Replan Validation

This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.
All variants use lower HF wait action prior gain `45.0s` so promotion is validated inside the full Freq-HRL lower-control loop.
Each native batch uses `3` shared-PPO replay update(s).
Runner workers: `12`.

| variant | seed | reward | wait | cv | score | upper decisions | gate replans | gate | promotion strength | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interval_only | 151 | -4974.360 | 10.0240 | 0.5679 | -11.1598 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| interval_only | 161 | -2806.211 | 4.4400 | 0.4427 | -5.3254 | 66.0 | 0.0 | 0.000 | 0.0000 | 4971 |
| interval_only | 171 | -16021.059 | 20.8140 | 0.3396 | -21.4932 | 66.0 | 0.0 | 0.000 | 0.8462 | 4970 |
| interval_only | 181 | -19828.767 | 14.4530 | 0.6218 | -15.6966 | 66.0 | 0.0 | 0.000 | 0.0000 | 4969 |
| interval_only | 191 | -4136.882 | 5.1140 | 0.4247 | -5.9634 | 66.0 | 0.0 | 0.000 | 1.0000 | 4969 |
| interval_only | 201 | -18943.941 | 5.2730 | 0.5776 | -6.4282 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| interval_only | 211 | -4636.787 | 11.6650 | 0.4461 | -12.5572 | 66.0 | 0.0 | 0.000 | 0.6923 | 4970 |
| interval_only | 221 | -18380.769 | 4.8810 | 0.5802 | -6.0414 | 66.0 | 0.0 | 0.000 | 0.0000 | 4972 |
| current_learned_gate | 151 | -4988.478 | 10.1810 | 0.4548 | -11.0906 | 66.0 | 2.0 | 0.982 | 0.0000 | 4970 |
| current_learned_gate | 161 | -2811.087 | 4.4400 | 0.4427 | -5.3254 | 66.0 | 2.0 | 0.982 | 0.0000 | 4971 |
| current_learned_gate | 171 | -16021.059 | 20.8140 | 0.3396 | -21.4932 | 66.0 | 0.0 | 0.000 | 0.8462 | 4970 |
| current_learned_gate | 181 | -19817.341 | 14.4460 | 0.7058 | -15.8576 | 66.0 | 2.0 | 0.982 | 0.0000 | 4969 |
| current_learned_gate | 191 | -3857.225 | 5.0630 | 0.4556 | -5.9742 | 66.0 | 2.0 | 0.982 | 1.0000 | 4969 |
| current_learned_gate | 201 | -18943.941 | 5.2730 | 0.5776 | -6.4282 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| current_learned_gate | 211 | -4854.391 | 12.6330 | 0.3867 | -13.4064 | 66.0 | 2.0 | 0.982 | 0.0769 | 4970 |
| current_learned_gate | 221 | -18384.747 | 4.8770 | 0.5907 | -6.0584 | 66.0 | 2.0 | 0.982 | 0.0000 | 4972 |
| safe_total1_gate | 151 | -5034.601 | 10.3650 | 0.5360 | -11.4370 | 66.0 | 1.0 | 0.982 | 0.0000 | 4970 |
| safe_total1_gate | 161 | -2810.010 | 4.4400 | 0.4427 | -5.3254 | 66.0 | 1.0 | 0.982 | 0.0000 | 4971 |
| safe_total1_gate | 171 | -16021.059 | 20.8140 | 0.3396 | -21.4932 | 66.0 | 0.0 | 0.000 | 0.8462 | 4970 |
| safe_total1_gate | 181 | -19831.327 | 14.4530 | 0.6218 | -15.6966 | 66.0 | 1.0 | 0.982 | 0.0000 | 4969 |
| safe_total1_gate | 191 | -4488.140 | 5.4310 | 0.4393 | -6.3096 | 66.0 | 1.0 | 0.982 | 1.0000 | 4969 |
| safe_total1_gate | 201 | -18943.941 | 5.2730 | 0.5776 | -6.4282 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| safe_total1_gate | 211 | -4667.335 | 11.9200 | 0.5243 | -12.9686 | 66.0 | 1.0 | 0.982 | 0.8286 | 4970 |
| safe_total1_gate | 221 | -18384.056 | 4.8830 | 0.5805 | -6.0440 | 66.0 | 1.0 | 0.982 | 0.0000 | 4972 |
| safe_strong_gate | 151 | -4974.360 | 10.0240 | 0.5679 | -11.1598 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| safe_strong_gate | 161 | -2806.211 | 4.4400 | 0.4427 | -5.3254 | 66.0 | 0.0 | 0.000 | 0.0000 | 4971 |
| safe_strong_gate | 171 | -16021.059 | 20.8140 | 0.3396 | -21.4932 | 66.0 | 0.0 | 0.000 | 0.8462 | 4970 |
| safe_strong_gate | 181 | -19828.767 | 14.4530 | 0.6218 | -15.6966 | 66.0 | 0.0 | 0.000 | 0.0000 | 4969 |
| safe_strong_gate | 191 | -4136.882 | 5.1140 | 0.4247 | -5.9634 | 66.0 | 0.0 | 0.000 | 1.0000 | 4969 |
| safe_strong_gate | 201 | -18943.941 | 5.2730 | 0.5776 | -6.4282 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| safe_strong_gate | 211 | -4636.787 | 11.6650 | 0.4461 | -12.5572 | 66.0 | 0.0 | 0.000 | 0.6923 | 4970 |
| safe_strong_gate | 221 | -18380.769 | 4.8810 | 0.5802 | -6.0414 | 66.0 | 0.0 | 0.000 | 0.0000 | 4972 |
| safe_lf_gate | 151 | -4974.360 | 10.0240 | 0.5679 | -11.1598 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| safe_lf_gate | 161 | -2806.211 | 4.4400 | 0.4427 | -5.3254 | 66.0 | 0.0 | 0.000 | 0.0000 | 4971 |
| safe_lf_gate | 171 | -16021.059 | 20.8140 | 0.3396 | -21.4932 | 66.0 | 0.0 | 0.000 | 0.8462 | 4970 |
| safe_lf_gate | 181 | -19828.767 | 14.4530 | 0.6218 | -15.6966 | 66.0 | 0.0 | 0.000 | 0.0000 | 4969 |
| safe_lf_gate | 191 | -4136.882 | 5.1140 | 0.4247 | -5.9634 | 66.0 | 0.0 | 0.000 | 1.0000 | 4969 |
| safe_lf_gate | 201 | -18943.941 | 5.2730 | 0.5776 | -6.4282 | 66.0 | 0.0 | 0.000 | 0.0000 | 4970 |
| safe_lf_gate | 211 | -4636.787 | 11.6650 | 0.4461 | -12.5572 | 66.0 | 0.0 | 0.000 | 0.6923 | 4970 |
| safe_lf_gate | 221 | -18380.769 | 4.8810 | 0.5802 | -6.0414 | 66.0 | 0.0 | 0.000 | 0.0000 | 4972 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| current_learned_gate_vs_interval_ep_reward | inconclusive | ep_reward | 8 | +6.3134 | -80.7843 | +103.5404 | 0.25 |
| current_learned_gate_vs_interval_avg_wait_min | not_supported | avg_wait_min | 8 | +0.1329 | -0.0145 | +0.3808 | 0.38 |
| current_learned_gate_vs_interval_score | not_supported | score | 8 | -0.1211 | -0.3399 | +0.0131 | 0.12 |
| current_learned_gate_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 8 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| safe_total1_gate_vs_interval_ep_reward | not_supported | ep_reward | 8 | -56.4616 | -143.5935 | -5.0243 | 0.00 |
| safe_total1_gate_vs_interval_avg_wait_min | not_supported | avg_wait_min | 8 | +0.1144 | +0.0321 | +0.2240 | 0.00 |
| safe_total1_gate_vs_interval_score | not_supported | score | 8 | -0.1297 | -0.2571 | -0.0347 | 0.00 |
| safe_total1_gate_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 8 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| safe_strong_gate_vs_interval_ep_reward | not_supported | ep_reward | 8 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| safe_strong_gate_vs_interval_avg_wait_min | not_supported | avg_wait_min | 8 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| safe_strong_gate_vs_interval_score | not_supported | score | 8 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| safe_strong_gate_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 8 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| safe_lf_gate_vs_interval_ep_reward | not_supported | ep_reward | 8 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| safe_lf_gate_vs_interval_avg_wait_min | not_supported | avg_wait_min | 8 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| safe_lf_gate_vs_interval_score | not_supported | score | 8 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| safe_lf_gate_vs_interval_upper_plan_decisions | not_supported | upper_plan_decisions | 8 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
