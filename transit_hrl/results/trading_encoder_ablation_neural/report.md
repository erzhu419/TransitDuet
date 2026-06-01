# Trading Encoder Ablation

Causal decomposer ablation for the same `freq_hrl` protocol.

- seeds: [1, 2]
- bars per seed: 80
- scenario: `persistent_shift`
- best Sharpe encoder: `neural_state_space` (6.842)

| encoder | return | Sharpe | max DD | turnover | PromotionDelay | UpperHFPower | LowerLFDrift | FocusScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ema | 0.0406 | 6.809 | 0.0012 | 1.56 | 0.5 | 0.0067 | 1.372 | 0.448 |
| adaptive_wavelet | 0.0409 | 6.842 | 0.0012 | 1.68 | 0.0 | 0.0076 | 1.311 | 0.390 |
| neural_state_space | 0.0410 | 6.842 | 0.0012 | 1.69 | 0.0 | 0.0077 | 1.302 | 0.458 |
