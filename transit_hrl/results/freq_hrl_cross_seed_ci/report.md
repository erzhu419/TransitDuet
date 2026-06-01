# Freq-HRL Cross-Seed CI Report

- checks: 30
- enough paired seeds/sources: 26
- supported checks: 24
- paper-ready checks: 15
- n_common range: 3 / 7.5 / 41

`paper_ready` requires supported status, enough pairs, positive improvement CI, and either sign-test p <= threshold or at least 10 pairs.

| check | status | n | delta CI95 | win | p | paper ready |
|---|---|---:|---:|---:|---:|---|
| transit_full_reward_vs_base | supported | 3 | +0.0565 [+0.0485, +0.0608] | 1.00 | 0.2500 | False |
| transit_full_wait_vs_base | supported | 3 | -0.0758 [-0.0815, -0.0651] | 1.00 | 0.2500 | False |
| transit_full_lower_lf_vs_base | supported | 3 | -0.0199 [-0.0199, -0.0198] | 1.00 | 0.2500 | False |
| transit_wait_credit_vs_no_wait | supported | 3 | -0.1251 [-0.1468, -0.1020] | 1.00 | 0.2500 | False |
| transit_learned_promotion_reward_vs_interval | supported | 10 | +0.0047 [+0.0016, +0.0076] | 0.70 | 0.3438 | True |
| transit_learned_promotion_wait_vs_interval | supported | 10 | -0.0079 [-0.0096, -0.0062] | 1.00 | 0.0020 | True |
| transit_learned_promotion_replans_vs_interval | supported | 10 | +20.1000 [+14.9000, +25.5000] | 1.00 | 0.0020 | True |
| transit_learned_promotion_raw_lf_vs_interval | supported | 10 | -0.0003 [-0.0003, -0.0003] | 1.00 | 0.0020 | True |
| transit_native_promotion_reward_vs_interval | inconclusive | 12 | +0.0168 [-11.2749, +16.2435] | 0.25 | 0.1460 | False |
| transit_native_promotion_wait_vs_interval | not_supported | 12 | +0.0092 [-0.0373, +0.0716] | 0.50 | 0.5078 | False |
| transit_native_promotion_replans_vs_interval | supported | 12 | +49.4167 [+36.4146, +62.0000] | 1.00 | 0.0005 | True |
| transit_native_learned_gate_reward_vs_interval | positive_mixed | 12 | +11.1923 [-9.1302, +33.6636] | 0.58 | 0.7744 | False |
| transit_native_learned_gate_wait_vs_interval | inconclusive | 12 | -0.0066 [-0.0523, +0.0438] | 0.33 | 1.0000 | False |
| transit_native_learned_gate_score_vs_interval | supported | 12 | +0.0385 [+0.0047, +0.0752] | 0.58 | 0.1797 | False |
| transit_native_learned_gate_replans_vs_interval | supported | 12 | +4.3333 [+3.2500, +5.5000] | 1.00 | 0.0005 | True |
| transit_native_learned_gate_gate_replans_vs_interval | supported | 12 | +15.6667 [+11.0000, +20.4167] | 1.00 | 0.0005 | True |
| transit_native_wait_credit_final_wait_vs_no_wait | supported | 5 | -8.5908 [-20.4276, -0.9074] | 0.80 | 0.3750 | False |
| transit_native_wait_credit_mean_wait_vs_no_wait | supported | 5 | -6.1649 [-16.1955, -0.4631] | 0.80 | 0.3750 | False |
| transit_native_wait_credit_reward_vs_no_wait | positive_mixed | 5 | +2037.3278 [-66.8598, +6165.1210] | 0.60 | 1.0000 | False |
| transit_native_wait_credit_score_vs_no_wait | supported | 5 | +8.6724 [+1.0542, +20.4807] | 0.80 | 0.3750 | False |
| transit_native_wait_credit_active_vs_no_wait | supported | 5 | +0.4500 [+0.4500, +0.4500] | 1.00 | 0.0625 | True |
| demand_nb_vs_fourier_mse | supported | 41 | -93.5937 [-139.1670, -48.3963] | 0.78 | 0.0004 | True |
| demand_nb_vs_fourier_mae | supported | 41 | -1.8545 [-2.7563, -0.9781] | 0.88 | 0.0000 | True |
| demand_nb_vs_fourier_poisson_nll_no_const | not_supported | 41 | +24.4245 [+13.9743, +36.8789] | 0.44 | 0.5327 | False |
| trading_constraint_lower_lf | supported | 5 | -1.0782 [-1.2590, -0.8371] | 1.00 | 0.0625 | True |
| trading_constraint_return_tradeoff | supported | 5 | -0.0003 [-0.0011, +0.0006] | 0.40 | 1.0000 | False |
| trading_constraint_raw_lower_lf | supported | 5 | -0.0000 [-0.0000, -0.0000] | 1.00 | 0.0625 | True |
| transit_constraint_lower_lf | supported | 5 | -0.3090 [-0.3382, -0.2834] | 1.00 | 0.0625 | True |
| transit_constraint_reward_tradeoff | supported | 5 | +0.0316 [+0.0308, +0.0322] | 1.00 | 0.0625 | True |
| transit_constraint_raw_lower_lf | supported | 5 | -0.0192 [-0.0193, -0.0192] | 1.00 | 0.0625 | True |
