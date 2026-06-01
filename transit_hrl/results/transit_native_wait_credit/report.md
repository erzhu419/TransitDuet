# Native Transit Wait-Credit Validation

This compares the native shared-PPO episode loop with and without frequency-attributed passenger wait credit.

| variant | seed | final wait | mean wait | final reward | final score | wait improvement | upper credit std | pax | samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_wait_credit | 31 | 6.2060 | 5.4000 | -2639.694 | -7.1502 | -2.2600 | 0.0000 | 0.0 | 4972 |
| no_wait_credit | 41 | 13.7290 | 7.9863 | -2582.097 | -14.5882 | -9.9360 | 0.0000 | 0.0 | 4970 |
| no_wait_credit | 51 | 43.2060 | 34.2543 | -17553.222 | -44.3650 | -13.9340 | 0.0000 | 0.0 | 4970 |
| no_wait_credit | 61 | 5.5550 | 5.1400 | -2949.197 | -6.6360 | -0.3730 | 0.0000 | 0.0 | 4970 |
| no_wait_credit | 71 | 6.5590 | 5.7660 | -2809.670 | -7.4438 | -2.4640 | 0.0000 | 0.0 | 4970 |
| native_wait_credit | 31 | 6.2070 | 5.4003 | -2667.580 | -7.1512 | -2.2610 | 0.4500 | 18554.7 | 4972 |
| native_wait_credit | 41 | 5.0250 | 4.9967 | -2540.636 | -6.0236 | -1.2170 | 0.4500 | 17794.7 | 4970 |
| native_wait_credit | 51 | 12.0610 | 8.2583 | -7281.452 | -13.0850 | -5.6710 | 0.4500 | 19269.7 | 4970 |
| native_wait_credit | 61 | 4.0420 | 4.6627 | -3086.084 | -4.8754 | +0.9660 | 0.4500 | 18287.0 | 4970 |
| native_wait_credit | 71 | 4.9660 | 4.4043 | -2771.489 | -5.6862 | -0.8740 | 0.4500 | 18232.3 | 4970 |

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| native_wait_credit_vs_no_wait_final_avg_wait_min | supported | final_avg_wait_min | 5 | -8.5908 | -20.4276 | -0.9074 | 0.80 |
| native_wait_credit_vs_no_wait_avg_wait_min_mean | supported | avg_wait_min_mean | 5 | -6.1649 | -16.1955 | -0.4631 | 0.80 |
| native_wait_credit_vs_no_wait_final_ep_reward | positive_mixed | final_ep_reward | 5 | +2037.3278 | -66.8598 | +6165.1210 | 0.60 |
| native_wait_credit_vs_no_wait_final_score | supported | final_score | 5 | +8.6724 | +1.0542 | +20.4807 | 0.80 |
| native_wait_credit_vs_no_wait_wait_improvement | supported | wait_improvement | 5 | +3.9820 | +0.8030 | +7.1108 | 0.80 |
| native_wait_credit_vs_no_wait_freq_wait_upper_credit_std | supported | freq_wait_upper_credit_std | 5 | +0.4500 | +0.4500 | +0.4500 | 1.00 |
