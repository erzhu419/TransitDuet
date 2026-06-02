# L2 Order-Book Matching Validation

- best Sharpe: `state_space` mode=market latency=0 (599.280)
- sources: `5`
- boundary: L2 market/passive-queue matching with latency and partial fills; best-level queue priority is approximated from L2 snapshots, not full L3 event replay

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| adaptive_wavelet_vs_ema_sharpe | positive_mixed | sharpe | 30 | +1.4779 | -1.3087 | +3.8865 | 0.50 |
| adaptive_wavelet_vs_ema_total_return | supported | total_return | 30 | +0.0000 | +0.0000 | +0.0000 | 0.57 |
| adaptive_wavelet_vs_ema_max_drawdown | supported | max_drawdown | 30 | -0.0000 | -0.0000 | -0.0000 | 0.67 |
| adaptive_wavelet_vs_ema_avg_slippage_bps | inconclusive | avg_slippage_bps | 30 | -0.0000 | -0.0000 | +0.0000 | 0.20 |
| adaptive_wavelet_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 30 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| adaptive_wavelet_vs_ema_fill_rate | not_supported | fill_rate | 30 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| neural_state_space_vs_ema_sharpe | supported | sharpe | 30 | +2.2925 | -0.0205 | +4.4852 | 0.57 |
| neural_state_space_vs_ema_total_return | supported | total_return | 30 | +0.0000 | +0.0000 | +0.0000 | 0.60 |
| neural_state_space_vs_ema_max_drawdown | supported | max_drawdown | 30 | -0.0000 | -0.0001 | -0.0000 | 0.67 |
| neural_state_space_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 30 | +0.0000 | -0.0000 | +0.0000 | 0.13 |
| neural_state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 30 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| neural_state_space_vs_ema_fill_rate | not_supported | fill_rate | 30 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| state_space_vs_ema_sharpe | supported | sharpe | 30 | +366.2083 | +299.1326 | +432.6357 | 0.97 |
| state_space_vs_ema_total_return | supported | total_return | 30 | +0.0011 | +0.0008 | +0.0015 | 0.87 |
| state_space_vs_ema_max_drawdown | supported | max_drawdown | 30 | -0.0012 | -0.0016 | -0.0009 | 0.93 |
| state_space_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 30 | +0.0016 | +0.0002 | +0.0031 | 0.43 |
| state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 30 | +0.0031 | +0.0003 | +0.0060 | 0.07 |
| state_space_vs_ema_fill_rate | not_supported | fill_rate | 30 | -0.0031 | -0.0060 | -0.0003 | 0.07 |
