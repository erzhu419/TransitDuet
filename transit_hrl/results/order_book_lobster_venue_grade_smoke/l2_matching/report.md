# L2 Order-Book Matching Validation

- best Sharpe: `adaptive_wavelet` mode=market latency=2 (209.410)
- sources: `1`
- boundary: L2 market/passive-queue matching with latency and partial fills; best-level queue priority is approximated from L2 snapshots, not full L3 event replay

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| adaptive_wavelet_vs_ema_sharpe | positive_mixed | sharpe | 4 | +30.8319 | -3.8094 | +85.7743 | 0.75 |
| adaptive_wavelet_vs_ema_total_return | not_supported | total_return | 4 | -0.0000 | -0.0003 | +0.0004 | 0.25 |
| adaptive_wavelet_vs_ema_max_drawdown | positive_mixed | max_drawdown | 4 | -0.0001 | -0.0005 | +0.0001 | 0.50 |
| adaptive_wavelet_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 4 | +0.0306 | +0.0000 | +0.0698 | 0.00 |
| adaptive_wavelet_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 4 | +0.0059 | +0.0000 | +0.0118 | 0.00 |
| adaptive_wavelet_vs_ema_fill_rate | not_supported | fill_rate | 4 | -0.0059 | -0.0118 | +0.0000 | 0.00 |
| neural_state_space_vs_ema_sharpe | not_supported | sharpe | 4 | -5.2460 | -37.0688 | +26.4206 | 0.25 |
| neural_state_space_vs_ema_total_return | not_supported | total_return | 4 | -0.0000 | -0.0002 | +0.0003 | 0.25 |
| neural_state_space_vs_ema_max_drawdown | positive_mixed | max_drawdown | 4 | -0.0001 | -0.0003 | +0.0001 | 0.50 |
| neural_state_space_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 4 | +0.0164 | +0.0000 | +0.0341 | 0.25 |
| neural_state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 4 | +0.0035 | +0.0000 | +0.0070 | 0.00 |
| neural_state_space_vs_ema_fill_rate | not_supported | fill_rate | 4 | -0.0035 | -0.0070 | +0.0000 | 0.00 |
| state_space_vs_ema_sharpe | not_supported | sharpe | 4 | -11.0506 | -120.0857 | +139.3313 | 0.25 |
| state_space_vs_ema_total_return | positive_mixed | total_return | 4 | +0.0006 | -0.0002 | +0.0015 | 0.75 |
| state_space_vs_ema_max_drawdown | positive_mixed | max_drawdown | 4 | -0.0007 | -0.0015 | +0.0000 | 0.75 |
| state_space_vs_ema_avg_slippage_bps | inconclusive | avg_slippage_bps | 4 | -0.0277 | -0.1024 | +0.0191 | 0.25 |
| state_space_vs_ema_partial_fill_rate | inconclusive | partial_fill_rate | 4 | -0.0070 | -0.0219 | +0.0010 | 0.25 |
| state_space_vs_ema_fill_rate | inconclusive | fill_rate | 4 | +0.0073 | +0.0000 | +0.0219 | 0.25 |
