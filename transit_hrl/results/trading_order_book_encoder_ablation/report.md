# Order-Book Freq-HRL Encoder Validation

- source: `synthetic_order_book_seed7`
- best Sharpe encoder: `state_space` (299.985)

| encoder | bars | return | Sharpe | max DD | turnover | promotions | spread bps | abs imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ema | 719 | 0.0005 | 220.026 | 0.0002 | 0.45 | 0 | 1.017 | 0.355 |
| state_space | 719 | 0.0040 | 299.985 | 0.0006 | 11.46 | 0 | 1.017 | 0.355 |
| adaptive_wavelet | 719 | 0.0006 | 220.823 | 0.0002 | 0.45 | 0 | 1.017 | 0.355 |
| neural_state_space | 719 | 0.0006 | 220.477 | 0.0002 | 0.47 | 0 | 1.017 | 0.355 |
