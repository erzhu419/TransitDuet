# Trading Policy Entry

- mode: `train`
- policy: `pg_linear`
- scenario: `persistent_shift`
- eval seeds: [31415, 27182, 16180, 11235, 4242]
- total return mean: 0.2495
- Sharpe mean: 15.993
- max drawdown mean: 0.0135
- turnover mean: 5.98
- leakage penalty mean: 1.7904

The `linear` policy is trained by cross-entropy policy search over shared frequency-routing coefficients. The `pg_linear` policy is trained by on-policy Gaussian REINFORCE over upper targets and lower execution speeds. Optional PG leakage flags add a policy-loss penalty and Lagrange-style constraint update using causal action-effect leakage. These are learned-policy validation paths, not full SAC/PPO implementations.

## Learned Parameters

| parameter | value |
|---|---:|
| upper_low | +1.0141 |
| upper_mid | +0.1963 |
| upper_high | -0.0215 |
| upper_promotion | +0.4000 |
| upper_position | -0.0434 |
| upper_bias | +0.0565 |
| lower_base_logit | +0.2109 |
| lower_align | +0.1744 |
| lower_energy | +0.0005 |
| upper_std | +0.2500 |
| lower_std | +0.2000 |
