# Public Market Encoder Ablation

- source: `yahoo_intraday`
- symbols: ['SPY', 'QQQ', 'IWM']
- date range: 2026-05-22T13:35:00+00:00 through 2026-05-29T20:00:00+00:00
- bar seconds: 300.0
- predictor: previous-bar log return only
- best Sharpe encoder: `adaptive_wavelet` (-9.220)
- best return encoder: `ema` (-0.0034)

| encoder | bars | return | Sharpe | max DD | turnover | promotions |
|---|---:|---:|---:|---:|---:|---:|
| ema | 390 | -0.0034 | -15.405 | 0.0034 | 3.89 | 0 |
| state_space | 390 | -0.0362 | -28.533 | 0.0362 | 36.11 | 0 |
| haar_wavelet | 390 | -0.0089 | -21.733 | 0.0089 | 9.63 | 0 |
| adaptive_wavelet | 390 | -0.0055 | -9.220 | 0.0058 | 6.32 | 0 |

This evaluates causal encoders on public market data. It is not investment advice and is not a production trading simulator.
