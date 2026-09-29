# Encoder Cross-Domain Matrix

Cross-domain encoder matrix assembled from existing experiment artifacts. Public market rows without paired seeds are marked summary_only; scheduler reruns should replace them with paired multi-seed or multi-window checks.

## Domain Summary

| domain | checks | supported | positive mixed | not supported | summary only | best status |
|---|---:|---:|---:|---:|---:|---|
| order_book_l2 | 18 | 8 | 1 | 8 | 0 | supported |
| order_book_l3 | 15 | 0 | 8 | 4 | 0 | positive_mixed |
| public_market_daily | 8 | 0 | 0 | 0 | 8 | summary_only |
| public_market_intraday | 12 | 0 | 0 | 0 | 12 | summary_only |
| trading_synthetic | 15 | 3 | 0 | 12 | 0 | supported |
| trading_synthetic_neural | 10 | 0 | 0 | 0 | 0 | not_supported |
| transit_real_demand | 12 | 4 | 4 | 4 | 0 | supported |
| transit_synthetic_demand | 6 | 3 | 0 | 3 | 0 | supported |

## Checks

| domain | check | status | metric | n | delta | CI95 low | CI95 high | win rate |
|---|---|---|---|---:|---:|---:|---:|---:|
| trading_synthetic | adaptive_wavelet_vs_ema_sharpe | not_supported | sharpe | 5 | -2.9876 | -3.3999 | -2.4549 | 0.00 |
| trading_synthetic | adaptive_wavelet_vs_ema_total_return | not_supported | total_return | 5 | -0.0358 | -0.0473 | -0.0256 | 0.00 |
| trading_synthetic | adaptive_wavelet_vs_ema_max_drawdown | not_supported | max_drawdown | 5 | +0.0172 | +0.0120 | +0.0232 | 0.00 |
| trading_synthetic | adaptive_wavelet_vs_ema_LowerLFDrift | supported | LowerLFDrift | 5 | -0.1130 | -0.1432 | -0.0880 | 1.00 |
| trading_synthetic | adaptive_wavelet_vs_ema_FocusScore | not_supported | FocusScore | 5 | -0.2444 | -0.3453 | -0.1435 | 0.00 |
| trading_synthetic | haar_wavelet_vs_ema_sharpe | not_supported | sharpe | 5 | -3.3509 | -4.1678 | -2.5339 | 0.00 |
| trading_synthetic | haar_wavelet_vs_ema_total_return | not_supported | total_return | 5 | -0.0458 | -0.0597 | -0.0319 | 0.00 |
| trading_synthetic | haar_wavelet_vs_ema_max_drawdown | not_supported | max_drawdown | 5 | +0.0145 | +0.0099 | +0.0200 | 0.00 |
| trading_synthetic | haar_wavelet_vs_ema_LowerLFDrift | supported | LowerLFDrift | 5 | -0.9222 | -1.0407 | -0.8162 | 1.00 |
| trading_synthetic | haar_wavelet_vs_ema_FocusScore | not_supported | FocusScore | 5 | -0.2754 | -0.3263 | -0.2325 | 0.00 |
| trading_synthetic | state_space_vs_ema_sharpe | not_supported | sharpe | 5 | -7.1319 | -8.4088 | -5.8743 | 0.00 |
| trading_synthetic | state_space_vs_ema_total_return | not_supported | total_return | 5 | -0.0827 | -0.1021 | -0.0632 | 0.00 |
| trading_synthetic | state_space_vs_ema_max_drawdown | not_supported | max_drawdown | 5 | +0.0402 | +0.0334 | +0.0477 | 0.00 |
| trading_synthetic | state_space_vs_ema_LowerLFDrift | supported | LowerLFDrift | 5 | -1.4981 | -1.5486 | -1.4433 | 1.00 |
| trading_synthetic | state_space_vs_ema_FocusScore | not_supported | FocusScore | 5 | -0.1833 | -0.2120 | -0.1533 | 0.00 |
| trading_synthetic_neural | adaptive_wavelet_vs_ema_sharpe | underpowered | sharpe | 2 | +0.0336 | +0.0210 | +0.0461 | 1.00 |
| trading_synthetic_neural | adaptive_wavelet_vs_ema_total_return | underpowered | total_return | 2 | +0.0003 | -0.0003 | +0.0009 | 0.50 |
| trading_synthetic_neural | adaptive_wavelet_vs_ema_max_drawdown | underpowered | max_drawdown | 2 | -0.0000 | -0.0000 | +0.0000 | 0.50 |
| trading_synthetic_neural | adaptive_wavelet_vs_ema_LowerLFDrift | underpowered | LowerLFDrift | 2 | -0.0610 | -0.0745 | -0.0476 | 1.00 |
| trading_synthetic_neural | adaptive_wavelet_vs_ema_FocusScore | underpowered | FocusScore | 2 | -0.0577 | -0.1954 | +0.0801 | 0.50 |
| trading_synthetic_neural | neural_state_space_vs_ema_sharpe | underpowered | sharpe | 2 | +0.0336 | +0.0301 | +0.0370 | 1.00 |
| trading_synthetic_neural | neural_state_space_vs_ema_total_return | underpowered | total_return | 2 | +0.0004 | -0.0004 | +0.0011 | 0.50 |
| trading_synthetic_neural | neural_state_space_vs_ema_max_drawdown | underpowered | max_drawdown | 2 | +0.0000 | -0.0000 | +0.0001 | 0.50 |
| trading_synthetic_neural | neural_state_space_vs_ema_LowerLFDrift | underpowered | LowerLFDrift | 2 | -0.0702 | -0.0872 | -0.0531 | 1.00 |
| trading_synthetic_neural | neural_state_space_vs_ema_FocusScore | underpowered | FocusScore | 2 | +0.0101 | -0.0641 | +0.0843 | 0.50 |
| public_market_daily | state_space_vs_ema_sharpe | summary_only | sharpe | 1 | -0.3401 | -0.3401 | -0.3401 | 0.00 |
| public_market_daily | state_space_vs_ema_total_return | summary_only | total_return | 1 | -0.4171 | -0.4171 | -0.4171 | 0.00 |
| public_market_daily | state_space_vs_ema_max_drawdown | summary_only | max_drawdown | 1 | -0.0150 | -0.0150 | -0.0150 | 1.00 |
| public_market_daily | state_space_vs_ema_turnover | summary_only | turnover | 1 | +247.7222 | +247.7222 | +247.7222 | 0.00 |
| public_market_daily | haar_wavelet_vs_ema_sharpe | summary_only | sharpe | 1 | +0.1897 | +0.1897 | +0.1897 | 1.00 |
| public_market_daily | haar_wavelet_vs_ema_total_return | summary_only | total_return | 1 | +0.3710 | +0.3710 | +0.3710 | 1.00 |
| public_market_daily | haar_wavelet_vs_ema_max_drawdown | summary_only | max_drawdown | 1 | +0.0292 | +0.0292 | +0.0292 | 0.00 |
| public_market_daily | haar_wavelet_vs_ema_turnover | summary_only | turnover | 1 | -71.1240 | -71.1240 | -71.1240 | 1.00 |
| public_market_intraday | state_space_vs_ema_sharpe | summary_only | sharpe | 1 | -13.1282 | -13.1282 | -13.1282 | 0.00 |
| public_market_intraday | state_space_vs_ema_total_return | summary_only | total_return | 1 | -0.0328 | -0.0328 | -0.0328 | 0.00 |
| public_market_intraday | state_space_vs_ema_max_drawdown | summary_only | max_drawdown | 1 | +0.0328 | +0.0328 | +0.0328 | 0.00 |
| public_market_intraday | state_space_vs_ema_turnover | summary_only | turnover | 1 | +32.2180 | +32.2180 | +32.2180 | 0.00 |
| public_market_intraday | haar_wavelet_vs_ema_sharpe | summary_only | sharpe | 1 | -6.3282 | -6.3282 | -6.3282 | 0.00 |
| public_market_intraday | haar_wavelet_vs_ema_total_return | summary_only | total_return | 1 | -0.0055 | -0.0055 | -0.0055 | 0.00 |
| public_market_intraday | haar_wavelet_vs_ema_max_drawdown | summary_only | max_drawdown | 1 | +0.0055 | +0.0055 | +0.0055 | 0.00 |
| public_market_intraday | haar_wavelet_vs_ema_turnover | summary_only | turnover | 1 | +5.7360 | +5.7360 | +5.7360 | 0.00 |
| public_market_intraday | adaptive_wavelet_vs_ema_sharpe | summary_only | sharpe | 1 | +6.1847 | +6.1847 | +6.1847 | 1.00 |
| public_market_intraday | adaptive_wavelet_vs_ema_total_return | summary_only | total_return | 1 | -0.0020 | -0.0020 | -0.0020 | 0.00 |
| public_market_intraday | adaptive_wavelet_vs_ema_max_drawdown | summary_only | max_drawdown | 1 | +0.0023 | +0.0023 | +0.0023 | 0.00 |
| public_market_intraday | adaptive_wavelet_vs_ema_turnover | summary_only | turnover | 1 | +2.4304 | +2.4304 | +2.4304 | 0.00 |
| order_book_l2 | adaptive_wavelet_vs_ema_sharpe | positive_mixed | sharpe | 30 | +1.4779 | -1.3087 | +3.8865 | 0.50 |
| order_book_l2 | adaptive_wavelet_vs_ema_total_return | supported | total_return | 30 | +0.0000 | +0.0000 | +0.0000 | 0.57 |
| order_book_l2 | adaptive_wavelet_vs_ema_max_drawdown | supported | max_drawdown | 30 | -0.0000 | -0.0000 | -0.0000 | 0.67 |
| order_book_l2 | adaptive_wavelet_vs_ema_avg_slippage_bps | inconclusive | avg_slippage_bps | 30 | -0.0000 | -0.0000 | +0.0000 | 0.20 |
| order_book_l2 | adaptive_wavelet_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 30 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| order_book_l2 | adaptive_wavelet_vs_ema_fill_rate | not_supported | fill_rate | 30 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| order_book_l2 | neural_state_space_vs_ema_sharpe | supported | sharpe | 30 | +2.2925 | -0.0205 | +4.4852 | 0.57 |
| order_book_l2 | neural_state_space_vs_ema_total_return | supported | total_return | 30 | +0.0000 | +0.0000 | +0.0000 | 0.60 |
| order_book_l2 | neural_state_space_vs_ema_max_drawdown | supported | max_drawdown | 30 | -0.0000 | -0.0001 | -0.0000 | 0.67 |
| order_book_l2 | neural_state_space_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 30 | +0.0000 | -0.0000 | +0.0000 | 0.13 |
| order_book_l2 | neural_state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 30 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| order_book_l2 | neural_state_space_vs_ema_fill_rate | not_supported | fill_rate | 30 | +0.0000 | +0.0000 | +0.0000 | 0.00 |
| order_book_l2 | state_space_vs_ema_sharpe | supported | sharpe | 30 | +366.2083 | +299.1326 | +432.6357 | 0.97 |
| order_book_l2 | state_space_vs_ema_total_return | supported | total_return | 30 | +0.0011 | +0.0008 | +0.0015 | 0.87 |
| order_book_l2 | state_space_vs_ema_max_drawdown | supported | max_drawdown | 30 | -0.0012 | -0.0016 | -0.0009 | 0.93 |
| order_book_l2 | state_space_vs_ema_avg_slippage_bps | not_supported | avg_slippage_bps | 30 | +0.0016 | +0.0002 | +0.0031 | 0.43 |
| order_book_l2 | state_space_vs_ema_partial_fill_rate | not_supported | partial_fill_rate | 30 | +0.0031 | +0.0003 | +0.0060 | 0.07 |
| order_book_l2 | state_space_vs_ema_fill_rate | not_supported | fill_rate | 30 | -0.0031 | -0.0060 | -0.0003 | 0.07 |
| order_book_l3 | adaptive_wavelet_vs_ema_sharpe | positive_mixed | sharpe | 10 | +21.5438 | -7.2826 | +53.6276 | 0.50 |
| order_book_l3 | adaptive_wavelet_vs_ema_total_return | not_supported | total_return | 10 | -0.0000 | -0.0000 | +0.0000 | 0.70 |
| order_book_l3 | adaptive_wavelet_vs_ema_max_drawdown | positive_mixed | max_drawdown | 10 | -0.0000 | -0.0000 | +0.0000 | 0.50 |
| order_book_l3 | adaptive_wavelet_vs_ema_fill_rate | positive_mixed | fill_rate | 10 | +0.0029 | -0.0033 | +0.0096 | 0.50 |
| order_book_l3 | adaptive_wavelet_vs_ema_avg_spread_capture_bps | not_supported | avg_spread_capture_bps | 10 | -0.0028 | -0.0074 | +0.0007 | 0.50 |
| order_book_l3 | neural_state_space_vs_ema_sharpe | not_supported | sharpe | 10 | -15.8827 | -75.4925 | +46.6949 | 0.50 |
| order_book_l3 | neural_state_space_vs_ema_total_return | positive_mixed | total_return | 10 | +0.0000 | -0.0000 | +0.0000 | 0.70 |
| order_book_l3 | neural_state_space_vs_ema_max_drawdown | inconclusive | max_drawdown | 10 | -0.0000 | -0.0000 | +0.0000 | 0.40 |
| order_book_l3 | neural_state_space_vs_ema_fill_rate | inconclusive | fill_rate | 10 | +0.0042 | -0.0088 | +0.0222 | 0.40 |
| order_book_l3 | neural_state_space_vs_ema_avg_spread_capture_bps | positive_mixed | avg_spread_capture_bps | 10 | +0.0042 | -0.0035 | +0.0130 | 0.50 |
| order_book_l3 | state_space_vs_ema_sharpe | not_supported | sharpe | 10 | -79.2900 | -135.2868 | -25.6541 | 0.20 |
| order_book_l3 | state_space_vs_ema_total_return | positive_mixed | total_return | 10 | +0.0000 | -0.0000 | +0.0000 | 0.50 |
| order_book_l3 | state_space_vs_ema_max_drawdown | positive_mixed | max_drawdown | 10 | -0.0000 | -0.0000 | -0.0000 | 0.60 |
| order_book_l3 | state_space_vs_ema_fill_rate | inconclusive | fill_rate | 10 | +0.0079 | -0.0096 | +0.0268 | 0.40 |
| order_book_l3 | state_space_vs_ema_avg_spread_capture_bps | positive_mixed | avg_spread_capture_bps | 10 | +0.0193 | -0.0038 | +0.0479 | 0.60 |
| transit_synthetic_demand | dynamic_harmonic_nb_vs_fourier | supported | mse | 5 | -2.2628 | -2.9443 | -1.6011 | 1.00 |
| transit_synthetic_demand | dynamic_harmonic_nb_vs_fourier | supported | mae | 5 | -0.1043 | -0.1559 | -0.0542 | 1.00 |
| transit_synthetic_demand | dynamic_harmonic_nb_vs_fourier | supported | poisson_nll_no_const | 5 | -0.0975 | -0.1293 | -0.0657 | 1.00 |
| transit_synthetic_demand | ema_vs_fourier | not_supported | mse | 5 | +5.3716 | +3.3430 | +7.5330 | 0.00 |
| transit_synthetic_demand | ema_vs_fourier | not_supported | mae | 5 | +1.0053 | +0.6793 | +1.3396 | 0.00 |
| transit_synthetic_demand | ema_vs_fourier | not_supported | poisson_nll_no_const | 5 | +0.1005 | +0.0575 | +0.1355 | 0.00 |
| transit_real_demand | adaptive_wavelet_vs_fourier | not_supported | mse | 12 | +0.1718 | +0.0777 | +0.2673 | 0.08 |
| transit_real_demand | adaptive_wavelet_vs_fourier | supported | mae | 12 | -0.0199 | -0.0317 | -0.0072 | 0.75 |
| transit_real_demand | adaptive_wavelet_vs_fourier | positive_mixed | poisson_nll_no_const | 12 | -0.0385 | -0.1246 | +0.0730 | 0.83 |
| transit_real_demand | dynamic_harmonic_nb_vs_fourier | positive_mixed | mse | 12 | -0.0795 | -0.1761 | +0.0075 | 0.75 |
| transit_real_demand | dynamic_harmonic_nb_vs_fourier | supported | mae | 12 | -0.0265 | -0.0356 | -0.0178 | 1.00 |
| transit_real_demand | dynamic_harmonic_nb_vs_fourier | supported | poisson_nll_no_const | 12 | -0.1466 | -0.1848 | -0.1104 | 1.00 |
| transit_real_demand | ema_vs_fourier | not_supported | mse | 12 | +0.3290 | +0.0552 | +0.5897 | 0.17 |
| transit_real_demand | ema_vs_fourier | not_supported | mae | 12 | +0.0038 | -0.0238 | +0.0291 | 0.33 |
| transit_real_demand | ema_vs_fourier | positive_mixed | poisson_nll_no_const | 12 | -0.0818 | -0.1973 | +0.0224 | 0.58 |
| transit_real_demand | neural_state_space_vs_fourier | not_supported | mse | 12 | +0.2333 | +0.0587 | +0.3857 | 0.25 |
| transit_real_demand | neural_state_space_vs_fourier | positive_mixed | mae | 12 | -0.0122 | -0.0276 | +0.0027 | 0.67 |
| transit_real_demand | neural_state_space_vs_fourier | supported | poisson_nll_no_const | 12 | -0.1048 | -0.1641 | -0.0481 | 0.83 |
