# Trading Encoder Ablation

Causal decomposer ablation for the same `freq_hrl` protocol.

- seeds: [42, 123, 456, 789, 2026]
- bars per seed: 720
- scenario: `persistent_shift`
- best Sharpe encoder: `ema` (16.062)

| encoder | return | Sharpe | max DD | turnover | PromotionDelay | UpperHFPower | LowerLFDrift | FocusScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ema | 0.2436 | 16.062 | 0.0093 | 5.76 | 23.4 | 0.0008 | 1.821 | 0.905 |
| fourier | 0.1299 | 7.011 | 0.0665 | 42.76 | -1.0 | 0.0317 | 0.286 | 0.630 |
| state_space | 0.1609 | 8.931 | 0.0494 | 37.74 | -1.0 | 0.0254 | 0.323 | 0.722 |
| haar_wavelet | 0.1978 | 12.712 | 0.0237 | 9.64 | 8.0 | 0.0044 | 0.899 | 0.630 |
| adaptive_wavelet | 0.2077 | 13.075 | 0.0264 | 7.70 | 213.8 | 0.0014 | 1.708 | 0.661 |
