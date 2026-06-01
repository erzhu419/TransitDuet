# Local Transit OD Demand Estimator Validation

- source: `transit_hrl/freq_transitduet/env/data/passenger_OD.xlsx`
- data path: copied TransitDuet OD spreadsheet, expanded from hourly OD counts into causal 5-minute AFC/APC-style bins
- best by MSE: `dynamic_harmonic_nb`

| method | series | MSE | MAE | Poisson NLL | delta MSE vs best |
|---|---:|---:|---:|---:|---:|
| dynamic_harmonic_nb | 12 | 26.9475 | 4.9587 | -0.1589 | +0.0000 |
| fourier | 12 | 27.0270 | 4.9853 | -0.0123 | +0.0795 |
| adaptive_wavelet | 12 | 27.1987 | 4.9654 | -0.0508 | +0.2513 |
| neural_state_space | 12 | 27.2602 | 4.9730 | -0.1171 | +0.3127 |
| ema | 12 | 27.3560 | 4.9891 | -0.0941 | +0.4085 |

## Paired Method Deltas

Deltas are `method - fourier`; lower is better for all listed metrics.

| comparison | metric | n | delta | CI95 low | CI95 high | win rate | status |
|---|---|---:|---:|---:|---:|---:|---|
| adaptive_wavelet_vs_fourier | mse | 12 | +0.1718 | +0.0777 | +0.2673 | 0.08 | not_supported |
| adaptive_wavelet_vs_fourier | mae | 12 | -0.0199 | -0.0317 | -0.0072 | 0.75 | supported |
| adaptive_wavelet_vs_fourier | poisson_nll_no_const | 12 | -0.0385 | -0.1246 | +0.0730 | 0.83 | positive_mixed |
| dynamic_harmonic_nb_vs_fourier | mse | 12 | -0.0795 | -0.1761 | +0.0075 | 0.75 | positive_mixed |
| dynamic_harmonic_nb_vs_fourier | mae | 12 | -0.0265 | -0.0356 | -0.0178 | 1.00 | supported |
| dynamic_harmonic_nb_vs_fourier | poisson_nll_no_const | 12 | -0.1466 | -0.1848 | -0.1104 | 1.00 | supported |
| ema_vs_fourier | mse | 12 | +0.3290 | +0.0552 | +0.5897 | 0.17 | not_supported |
| ema_vs_fourier | mae | 12 | +0.0038 | -0.0238 | +0.0291 | 0.33 | not_supported |
| ema_vs_fourier | poisson_nll_no_const | 12 | -0.0818 | -0.1973 | +0.0224 | 0.58 | positive_mixed |
| neural_state_space_vs_fourier | mse | 12 | +0.2333 | +0.0587 | +0.3857 | 0.25 | not_supported |
| neural_state_space_vs_fourier | mae | 12 | -0.0122 | -0.0276 | +0.0027 | 0.67 | positive_mixed |
| neural_state_space_vs_fourier | poisson_nll_no_const | 12 | -0.1048 | -0.1641 | -0.0481 | 0.83 | supported |
