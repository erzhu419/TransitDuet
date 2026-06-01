# Freq-HRL Cross-Seed CI Report

- checks: 20
- enough paired seeds/sources: 16
- supported checks: 18
- paper-ready checks: 13
- n_common range: 3 / 6.5 / 17

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
| transit_native_promotion_reward_vs_interval | positive_mixed | 8 | +331.9682 [-4100.6044, +4771.6282] | 0.62 | 0.7266 | False |
| transit_native_promotion_wait_vs_interval | positive_mixed | 8 | -4.4798 [-13.1886, +0.8364] | 0.62 | 0.7266 | False |
| transit_native_promotion_replans_vs_interval | supported | 8 | +54.8750 [+39.5000, +69.8844] | 1.00 | 0.0078 | True |
| demand_nb_vs_fourier_mse | supported | 17 | -0.7217 [-1.2633, -0.2617] | 0.82 | 0.0127 | True |
| demand_nb_vs_fourier_mae | supported | 17 | -0.0494 [-0.0759, -0.0296] | 1.00 | 0.0000 | True |
| demand_nb_vs_fourier_poisson_nll_no_const | supported | 17 | -0.1322 [-0.1620, -0.1047] | 1.00 | 0.0000 | True |
| trading_constraint_lower_lf | supported | 5 | -1.0782 [-1.2590, -0.8371] | 1.00 | 0.0625 | True |
| trading_constraint_return_tradeoff | supported | 5 | -0.0003 [-0.0011, +0.0006] | 0.40 | 1.0000 | False |
| trading_constraint_raw_lower_lf | supported | 5 | -0.0000 [-0.0000, -0.0000] | 1.00 | 0.0625 | True |
| transit_constraint_lower_lf | supported | 5 | -0.3090 [-0.3382, -0.2834] | 1.00 | 0.0625 | True |
| transit_constraint_reward_tradeoff | supported | 5 | +0.0316 [+0.0308, +0.0322] | 1.00 | 0.0625 | True |
| transit_constraint_raw_lower_lf | supported | 5 | -0.0192 [-0.0193, -0.0192] | 1.00 | 0.0625 | True |
