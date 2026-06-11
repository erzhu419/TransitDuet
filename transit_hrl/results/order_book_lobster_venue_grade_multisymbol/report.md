# Order-Book Large Replay Manifest Validation

- manifest entries: `6`
- used entries: `6`
- L2 files: `3`
- L3 files: `3`
- real/venue-grade L2 files: `3`
- real/venue-grade L3 files: `3`
- venue-grade paired L2/L3 sessions: `3`
- venue-grade schema-ready L2/L3 sessions: `3`
- schema-ready L2 files: `3`
- schema-ready L3 files: `3`
- source quality: `venue_grade_ready`
- venue-grade claim status: `supported`
- venue-grade required: `True`
- missing entries: `0`
- boundary: Manifest-driven real/fixture L2/L3 replay. L2 uses market/passive matching with a best-level queue-priority proxy; L3 uses FIFO add/cancel/trade event replay for agent passive orders. Venue-grade claims require real exchange feeds in the manifest.

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| l2_adaptive_wavelet_vs_ema_sharpe | not_supported | sharpe | 12 | -7.0963 | -245.8319 | +213.6134 | 0.67 |
| l2_adaptive_wavelet_vs_ema_total_return | not_supported | total_return | 12 | -0.0000 | -0.0005 | +0.0004 | 0.25 |
| l2_adaptive_wavelet_vs_ema_max_drawdown | positive_mixed | max_drawdown | 12 | -0.0001 | -0.0005 | +0.0004 | 0.67 |
| l2_adaptive_wavelet_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 12 | +0.0098 | -0.0030 | +0.0276 | 0.25 |
| l2_adaptive_wavelet_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 12 | +0.0021 | -0.0005 | +0.0051 | 0.08 |
| l2_adaptive_wavelet_vs_ema_fill_rate | not_supported | fill_rate | 12 | -0.0021 | -0.0051 | +0.0005 | 0.08 |
| l2_neural_state_space_vs_ema_sharpe | not_supported | sharpe | 12 | -26.4263 | -270.7543 | +197.9512 | 0.33 |
| l2_neural_state_space_vs_ema_total_return | inconclusive | total_return | 12 | +0.0000 | -0.0004 | +0.0006 | 0.33 |
| l2_neural_state_space_vs_ema_max_drawdown | inconclusive | max_drawdown | 12 | -0.0001 | -0.0006 | +0.0004 | 0.42 |
| l2_neural_state_space_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 12 | +0.0045 | -0.0030 | +0.0137 | 0.42 |
| l2_neural_state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 12 | +0.0013 | -0.0005 | +0.0031 | 0.17 |
| l2_neural_state_space_vs_ema_fill_rate | not_supported | fill_rate | 12 | -0.0013 | -0.0031 | +0.0005 | 0.17 |
| l2_state_space_vs_ema_sharpe | positive_mixed | sharpe | 12 | +2.5770 | -53.1498 | +59.5151 | 0.50 |
| l2_state_space_vs_ema_total_return | supported | total_return | 12 | +0.0004 | +0.0001 | +0.0008 | 0.75 |
| l2_state_space_vs_ema_max_drawdown | supported | max_drawdown | 12 | -0.0004 | -0.0008 | -0.0001 | 0.83 |
| l2_state_space_vs_ema_avg_slippage_bps | positive_mixed | avg_slippage_bps | 12 | -0.0175 | -0.0440 | +0.0015 | 0.58 |
| l2_state_space_vs_ema_partial_fill_rate | inconclusive | partial_fill_rate | 12 | -0.0051 | -0.0114 | +0.0001 | 0.25 |
| l2_state_space_vs_ema_fill_rate | inconclusive | fill_rate | 12 | +0.0052 | +0.0000 | +0.0115 | 0.25 |
| l3_adaptive_wavelet_vs_ema_sharpe | not_supported | sharpe | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_adaptive_wavelet_vs_ema_total_return | not_supported | total_return | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_adaptive_wavelet_vs_ema_max_drawdown | not_supported | max_drawdown | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_adaptive_wavelet_vs_ema_fill_rate | not_supported | fill_rate | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_adaptive_wavelet_vs_ema_avg_spread_capture_bps | not_supported | avg_spread_capture_bps | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_neural_state_space_vs_ema_sharpe | not_supported | sharpe | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_neural_state_space_vs_ema_total_return | not_supported | total_return | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_neural_state_space_vs_ema_max_drawdown | not_supported | max_drawdown | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_neural_state_space_vs_ema_fill_rate | not_supported | fill_rate | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_neural_state_space_vs_ema_avg_spread_capture_bps | not_supported | avg_spread_capture_bps | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_state_space_vs_ema_sharpe | not_supported | sharpe | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_state_space_vs_ema_total_return | not_supported | total_return | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_state_space_vs_ema_max_drawdown | not_supported | max_drawdown | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_state_space_vs_ema_fill_rate | not_supported | fill_rate | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_state_space_vs_ema_avg_spread_capture_bps | not_supported | avg_spread_capture_bps | 3 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
