# Trading Policy Entry

- mode: `train`
- policy: `pg_linear`
- scenario: `persistent_shift`
- eval seeds: [31415, 27182, 16180, 11235, 4242]
- total return mean: 0.2487
- Sharpe mean: 15.915
- max drawdown mean: 0.0138
- turnover mean: 6.41

The `linear` policy is trained by cross-entropy policy search over shared frequency-routing coefficients. The `pg_linear` policy is trained by on-policy Gaussian REINFORCE over upper targets and lower execution speeds. These are learned-policy validation paths, not full SAC/PPO implementations.

## Learned Parameters

| parameter | value |
|---|---:|
| upper_low | +1.0133 |
| upper_mid | +0.1972 |
| upper_high | -0.0123 |
| upper_promotion | +0.4000 |
| upper_position | -0.0394 |
| upper_bias | +0.0573 |
| lower_base_logit | +0.2008 |
| lower_align | +0.1771 |
| lower_energy | +0.0006 |
| upper_std | +0.2500 |
| lower_std | +0.2000 |
