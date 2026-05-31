# Trading Policy Entry

- mode: `train`
- policy: `linear`
- scenario: `persistent_shift`
- eval seeds: [31415, 27182, 16180, 11235, 4242]
- total return mean: 0.2838
- Sharpe mean: 16.016
- max drawdown mean: 0.0166
- turnover mean: 6.78

The `linear` policy is trained by cross-entropy policy search over shared frequency-routing coefficients. It is a lightweight learned-policy validation path for the Freq-HRL protocol, not a full SAC/PPO implementation.

## Learned Parameters

| parameter | value |
|---|---:|
| upper_low | +1.9654 |
| upper_mid | -0.0654 |
| upper_promotion | +0.7525 |
| upper_bias | +0.2234 |
| lower_base | +0.0582 |
| lower_align | +0.7365 |
| residual_high | +0.0574 |
