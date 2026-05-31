# Public Market Encoder Ablation

- source: `csv`
- symbols: ['SPY', 'QQQ', 'IWM']
- date range: 2016-06-01 through 2026-05-29
- predictor: previous-bar log return only
- best Sharpe encoder: `haar_wavelet` (0.596)
- best return encoder: `haar_wavelet` (0.7667)

| encoder | bars | return | Sharpe | max DD | turnover | promotions |
|---|---:|---:|---:|---:|---:|---:|
| ema | 1500 | 0.3957 | 0.406 | 0.2790 | 106.85 | 246 |
| state_space | 1500 | -0.0214 | 0.066 | 0.2640 | 354.57 | 699 |
| haar_wavelet | 1500 | 0.7667 | 0.596 | 0.3082 | 35.72 | 1471 |

This evaluates causal encoders on public market data. It is not investment advice and is not a production trading simulator.
