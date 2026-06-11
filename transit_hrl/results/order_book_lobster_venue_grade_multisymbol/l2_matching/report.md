# L2 Order-Book Matching Validation

- best Sharpe: `ema` mode=market latency=2 (510.708)
- sources: `3`
- boundary: L2 market/passive-queue matching with latency and partial fills; best-level queue priority is approximated from L2 snapshots, not full L3 event replay

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| adaptive_wavelet_vs_ema_sharpe | not_supported | sharpe | 12 | -7.0963 | -245.8319 | +213.6134 | 0.67 |
| adaptive_wavelet_vs_ema_total_return | not_supported | total_return | 12 | -0.0000 | -0.0005 | +0.0004 | 0.25 |
| adaptive_wavelet_vs_ema_max_drawdown | positive_mixed | max_drawdown | 12 | -0.0001 | -0.0005 | +0.0004 | 0.67 |
| adaptive_wavelet_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 12 | +0.0098 | -0.0030 | +0.0276 | 0.25 |
| adaptive_wavelet_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 12 | +0.0021 | -0.0005 | +0.0051 | 0.08 |
| adaptive_wavelet_vs_ema_fill_rate | not_supported | fill_rate | 12 | -0.0021 | -0.0051 | +0.0005 | 0.08 |
| neural_state_space_vs_ema_sharpe | not_supported | sharpe | 12 | -26.4263 | -270.7543 | +197.9512 | 0.33 |
| neural_state_space_vs_ema_total_return | inconclusive | total_return | 12 | +0.0000 | -0.0004 | +0.0006 | 0.33 |
| neural_state_space_vs_ema_max_drawdown | inconclusive | max_drawdown | 12 | -0.0001 | -0.0006 | +0.0004 | 0.42 |
| neural_state_space_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 12 | +0.0045 | -0.0030 | +0.0137 | 0.42 |
| neural_state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 12 | +0.0013 | -0.0005 | +0.0031 | 0.17 |
| neural_state_space_vs_ema_fill_rate | not_supported | fill_rate | 12 | -0.0013 | -0.0031 | +0.0005 | 0.17 |
| state_space_vs_ema_sharpe | positive_mixed | sharpe | 12 | +2.5770 | -53.1498 | +59.5151 | 0.50 |
| state_space_vs_ema_total_return | supported | total_return | 12 | +0.0004 | +0.0001 | +0.0008 | 0.75 |
| state_space_vs_ema_max_drawdown | supported | max_drawdown | 12 | -0.0004 | -0.0008 | -0.0001 | 0.83 |
| state_space_vs_ema_avg_slippage_bps | positive_mixed | avg_slippage_bps | 12 | -0.0175 | -0.0440 | +0.0015 | 0.58 |
| state_space_vs_ema_partial_fill_rate | inconclusive | partial_fill_rate | 12 | -0.0051 | -0.0114 | +0.0001 | 0.25 |
| state_space_vs_ema_fill_rate | inconclusive | fill_rate | 12 | +0.0052 | +0.0000 | +0.0115 | 0.25 |
