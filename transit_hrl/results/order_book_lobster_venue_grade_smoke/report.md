# Order-Book Large Replay Manifest Validation

- manifest entries: `2`
- used entries: `2`
- L2 files: `1`
- L3 files: `1`
- real/venue-grade L2 files: `1`
- real/venue-grade L3 files: `1`
- venue-grade paired L2/L3 sessions: `1`
- venue-grade schema-ready L2/L3 sessions: `1`
- schema-ready L2 files: `1`
- schema-ready L3 files: `1`
- source quality: `venue_grade_ready`
- venue-grade claim status: `supported`
- venue-grade required: `True`
- missing entries: `0`
- boundary: Manifest-driven real/fixture L2/L3 replay. L2 uses market/passive matching with a best-level queue-priority proxy; L3 uses FIFO add/cancel/trade event replay for agent passive orders. Venue-grade claims require real exchange feeds in the manifest.

| check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---:|---:|---:|---:|---:|
| l2_adaptive_wavelet_vs_ema_sharpe | positive_mixed | sharpe | 4 | +30.8319 | -3.8094 | +85.7743 | 0.75 |
| l2_adaptive_wavelet_vs_ema_total_return | not_supported | total_return | 4 | -0.0000 | -0.0003 | +0.0004 | 0.25 |
| l2_adaptive_wavelet_vs_ema_max_drawdown | positive_mixed | max_drawdown | 4 | -0.0001 | -0.0005 | +0.0001 | 0.50 |
| l2_adaptive_wavelet_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 4 | +0.0306 | +0.0000 | +0.0698 | 0.00 |
| l2_adaptive_wavelet_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 4 | +0.0059 | +0.0000 | +0.0118 | 0.00 |
| l2_adaptive_wavelet_vs_ema_fill_rate | not_supported | fill_rate | 4 | -0.0059 | -0.0118 | +0.0000 | 0.00 |
| l2_neural_state_space_vs_ema_sharpe | not_supported | sharpe | 4 | -5.2460 | -37.0688 | +26.4206 | 0.25 |
| l2_neural_state_space_vs_ema_total_return | not_supported | total_return | 4 | -0.0000 | -0.0002 | +0.0003 | 0.25 |
| l2_neural_state_space_vs_ema_max_drawdown | positive_mixed | max_drawdown | 4 | -0.0001 | -0.0003 | +0.0001 | 0.50 |
| l2_neural_state_space_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 4 | +0.0164 | +0.0000 | +0.0341 | 0.25 |
| l2_neural_state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 4 | +0.0035 | +0.0000 | +0.0070 | 0.00 |
| l2_neural_state_space_vs_ema_fill_rate | not_supported | fill_rate | 4 | -0.0035 | -0.0070 | +0.0000 | 0.00 |
| l2_state_space_vs_ema_sharpe | not_supported | sharpe | 4 | -11.0506 | -120.0857 | +139.3313 | 0.25 |
| l2_state_space_vs_ema_total_return | positive_mixed | total_return | 4 | +0.0006 | -0.0002 | +0.0015 | 0.75 |
| l2_state_space_vs_ema_max_drawdown | positive_mixed | max_drawdown | 4 | -0.0007 | -0.0015 | +0.0000 | 0.75 |
| l2_state_space_vs_ema_avg_slippage_bps | inconclusive | avg_slippage_bps | 4 | -0.0277 | -0.1024 | +0.0191 | 0.25 |
| l2_state_space_vs_ema_partial_fill_rate | inconclusive | partial_fill_rate | 4 | -0.0070 | -0.0219 | +0.0010 | 0.25 |
| l2_state_space_vs_ema_fill_rate | inconclusive | fill_rate | 4 | +0.0073 | +0.0000 | +0.0219 | 0.25 |
| l3_adaptive_wavelet_vs_ema_sharpe | not_supported | sharpe | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_adaptive_wavelet_vs_ema_total_return | not_supported | total_return | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_adaptive_wavelet_vs_ema_max_drawdown | not_supported | max_drawdown | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_adaptive_wavelet_vs_ema_fill_rate | not_supported | fill_rate | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_adaptive_wavelet_vs_ema_avg_spread_capture_bps | not_supported | avg_spread_capture_bps | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_neural_state_space_vs_ema_sharpe | not_supported | sharpe | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_neural_state_space_vs_ema_total_return | not_supported | total_return | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_neural_state_space_vs_ema_max_drawdown | not_supported | max_drawdown | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_neural_state_space_vs_ema_fill_rate | not_supported | fill_rate | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_neural_state_space_vs_ema_avg_spread_capture_bps | not_supported | avg_spread_capture_bps | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_state_space_vs_ema_sharpe | not_supported | sharpe | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_state_space_vs_ema_total_return | not_supported | total_return | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_state_space_vs_ema_max_drawdown | not_supported | max_drawdown | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_state_space_vs_ema_fill_rate | not_supported | fill_rate | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| l3_state_space_vs_ema_avg_spread_capture_bps | not_supported | avg_spread_capture_bps | 1 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
