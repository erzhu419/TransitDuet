# Order-Book Depth Stress Validation

- best Sharpe: `adaptive_wavelet` under `stale_book` (243.483)
- boundary: synthetic or CSV L1/L2 stress validation, not a full exchange matching engine

## Paired Checks

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| adaptive_wavelet_vs_ema_sharpe | positive_mixed | sharpe | 25 | +0.2807 | -0.3942 | +0.9004 | 0.68 |
| adaptive_wavelet_vs_ema_total_return | not_supported | total_return | 25 | -0.0000 | -0.0000 | -0.0000 | 0.40 |
| adaptive_wavelet_vs_ema_max_drawdown | supported | max_drawdown | 25 | -0.0000 | -0.0000 | -0.0000 | 0.68 |
| adaptive_wavelet_vs_ema_turnover | not_supported | turnover | 25 | +0.0011 | +0.0006 | +0.0016 | 0.00 |
| neural_state_space_vs_ema_sharpe | not_supported | sharpe | 25 | -1.1349 | -2.4475 | +0.0254 | 0.48 |
| neural_state_space_vs_ema_total_return | not_supported | total_return | 25 | -0.0000 | -0.0000 | -0.0000 | 0.04 |
| neural_state_space_vs_ema_max_drawdown | supported | max_drawdown | 25 | -0.0000 | -0.0000 | -0.0000 | 0.84 |
| neural_state_space_vs_ema_turnover | not_supported | turnover | 25 | +0.0055 | +0.0050 | +0.0061 | 0.00 |
| state_space_vs_ema_sharpe | not_supported | sharpe | 25 | -71.3384 | -128.2308 | -16.8018 | 0.40 |
| state_space_vs_ema_total_return | not_supported | total_return | 25 | -0.0003 | -0.0004 | -0.0001 | 0.28 |
| state_space_vs_ema_max_drawdown | not_supported | max_drawdown | 25 | +0.0003 | +0.0001 | +0.0004 | 0.28 |
| state_space_vs_ema_turnover | not_supported | turnover | 25 | +3.4668 | +3.4136 | +3.5239 | 0.00 |
