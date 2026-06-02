# L3 Order-Event Replay Validation

- best Sharpe: `neural_state_space` (-185.029)
- sources: `10`
- boundary: L3 FIFO queue replay with add/cancel/trade events; committed run uses synthetic tapes unless CSVs are supplied

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| adaptive_wavelet_vs_ema_sharpe | positive_mixed | sharpe | 10 | +21.5438 | -7.2826 | +53.6276 | 0.50 |
| adaptive_wavelet_vs_ema_total_return | not_supported | total_return | 10 | -0.0000 | -0.0000 | +0.0000 | 0.70 |
| adaptive_wavelet_vs_ema_max_drawdown | positive_mixed | max_drawdown | 10 | -0.0000 | -0.0000 | +0.0000 | 0.50 |
| adaptive_wavelet_vs_ema_fill_rate | positive_mixed | fill_rate | 10 | +0.0029 | -0.0033 | +0.0096 | 0.50 |
| adaptive_wavelet_vs_ema_avg_spread_capture_bps | not_supported | avg_spread_capture_bps | 10 | -0.0028 | -0.0074 | +0.0007 | 0.50 |
| neural_state_space_vs_ema_sharpe | not_supported | sharpe | 10 | -15.8827 | -75.4925 | +46.6949 | 0.50 |
| neural_state_space_vs_ema_total_return | positive_mixed | total_return | 10 | +0.0000 | -0.0000 | +0.0000 | 0.70 |
| neural_state_space_vs_ema_max_drawdown | inconclusive | max_drawdown | 10 | -0.0000 | -0.0000 | +0.0000 | 0.40 |
| neural_state_space_vs_ema_fill_rate | inconclusive | fill_rate | 10 | +0.0042 | -0.0088 | +0.0222 | 0.40 |
| neural_state_space_vs_ema_avg_spread_capture_bps | positive_mixed | avg_spread_capture_bps | 10 | +0.0042 | -0.0035 | +0.0130 | 0.50 |
| state_space_vs_ema_sharpe | not_supported | sharpe | 10 | -79.2900 | -135.2868 | -25.6541 | 0.20 |
| state_space_vs_ema_total_return | positive_mixed | total_return | 10 | +0.0000 | -0.0000 | +0.0000 | 0.50 |
| state_space_vs_ema_max_drawdown | positive_mixed | max_drawdown | 10 | -0.0000 | -0.0000 | -0.0000 | 0.60 |
| state_space_vs_ema_fill_rate | inconclusive | fill_rate | 10 | +0.0079 | -0.0096 | +0.0268 | 0.40 |
| state_space_vs_ema_avg_spread_capture_bps | positive_mixed | avg_spread_capture_bps | 10 | +0.0193 | -0.0038 | +0.0479 | 0.60 |
