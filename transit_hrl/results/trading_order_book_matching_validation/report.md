# L2 Order-Book Matching Validation

- best Sharpe: `state_space` latency=0 (599.280)
- sources: `5`
- boundary: L2 market-order matching with latency and partial fills; no exchange queue priority

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| adaptive_wavelet_vs_ema_sharpe | inconclusive | sharpe | 15 | +1.0908 | -0.2300 | +2.4692 | 0.47 |
| adaptive_wavelet_vs_ema_total_return | supported | total_return | 15 | +0.0000 | +0.0000 | +0.0000 | 0.47 |
| adaptive_wavelet_vs_ema_max_drawdown | supported | max_drawdown | 15 | -0.0000 | -0.0000 | -0.0000 | 0.67 |
| adaptive_wavelet_vs_ema_avg_slippage_bps | inconclusive | avg_slippage_bps | 15 | -0.0000 | -0.0000 | +0.0000 | 0.40 |
| adaptive_wavelet_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 15 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| neural_state_space_vs_ema_sharpe | inconclusive | sharpe | 15 | +0.8548 | -0.8653 | +2.4053 | 0.47 |
| neural_state_space_vs_ema_total_return | supported | total_return | 15 | +0.0000 | +0.0000 | +0.0000 | 0.60 |
| neural_state_space_vs_ema_max_drawdown | supported | max_drawdown | 15 | -0.0000 | -0.0001 | -0.0000 | 0.67 |
| neural_state_space_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 15 | +0.0000 | -0.0000 | +0.0000 | 0.27 |
| neural_state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 15 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| state_space_vs_ema_sharpe | supported | sharpe | 15 | +483.0914 | +407.1948 | +559.5459 | 1.00 |
| state_space_vs_ema_total_return | supported | total_return | 15 | +0.0012 | +0.0006 | +0.0018 | 0.87 |
| state_space_vs_ema_max_drawdown | supported | max_drawdown | 15 | -0.0012 | -0.0017 | -0.0006 | 0.93 |
| state_space_vs_ema_avg_slippage_bps | positive_mixed | avg_slippage_bps | 15 | -0.0000 | -0.0000 | +0.0000 | 0.53 |
| state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 15 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
