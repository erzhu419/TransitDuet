# Trading Policy Entry

- mode: `train`
- policy: `ac_linear`
- scenario: `persistent_shift`
- eval seeds: [31415, 27182, 16180, 11235, 4242]
- total return mean: 0.2804
- Sharpe mean: 15.528
- max drawdown mean: 0.0162
- turnover mean: 6.49
- leakage penalty mean: 1.6475

The `linear` policy is trained by cross-entropy policy search over shared frequency-routing coefficients. The `pg_linear` policy is trained by on-policy Gaussian REINFORCE over upper targets and lower execution speeds. The `ac_linear` policy uses separated upper low-frequency and lower high-frequency TD(0) critics to train the same actor with bootstrapped advantages. Optional leakage flags add a policy-loss penalty and Lagrange-style constraint update using causal action-effect leakage.

## Learned Parameters

| parameter | value |
|---|---:|
| actor.upper_low | +1.1316 |
| actor.upper_mid | +0.1967 |
| actor.upper_high | -0.0039 |
| actor.upper_promotion | +0.4000 |
| actor.upper_position | +0.2117 |
| actor.upper_bias | +0.2161 |
| actor.lower_base_logit | -0.0566 |
| actor.lower_align | +0.2038 |
| actor.lower_energy | -0.0669 |
| actor.upper_std | +0.2500 |
| actor.lower_std | +0.2000 |
| upper_value[0] | -0.0002 |
| upper_value[1] | -0.0004 |
| upper_value[2] | -0.0000 |
| upper_value[3] | -0.0001 |
| upper_value[4] | -0.0000 |
| upper_value[5] | -0.0004 |
| lower_value[0] | -0.0005 |
| lower_value[1] | -0.0000 |
| lower_value[2] | -0.0005 |
| lower_value[3] | -0.0006 |
| lower_value[4] | -0.0021 |
