# Transit Demand Estimator Validation

- best by MSE: `dynamic_harmonic_nb`
- `dynamic_harmonic_nb` uses the count-calibrated online Newton path: learning_rate=1.0, ridge=0.05, NB dispersion=1000.0.

| method | seeds | MSE | MAE | Poisson NLL | delta MSE vs best |
|---|---:|---:|---:|---:|---:|
| dynamic_harmonic_nb | 5 | 6.9967 | 2.2135 | -39.7445 | +0.0000 |
| fourier | 5 | 9.2595 | 2.3179 | -39.6470 | +2.2628 |
| ema | 5 | 14.6311 | 3.3232 | -39.5465 | +7.6344 |

## Paired Method Deltas

Deltas are `method - fourier`; lower is better for all listed metrics.

| comparison | metric | n | delta | CI95 low | CI95 high | win rate | sign p | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_harmonic_nb_vs_fourier | mse | 5 | -2.2628 | -2.9443 | -1.6011 | 1.00 | 0.0625 | supported |
| dynamic_harmonic_nb_vs_fourier | mae | 5 | -0.1043 | -0.1559 | -0.0542 | 1.00 | 0.0625 | supported |
| dynamic_harmonic_nb_vs_fourier | poisson_nll_no_const | 5 | -0.0975 | -0.1293 | -0.0657 | 1.00 | 0.0625 | supported |
| ema_vs_fourier | mse | 5 | +5.3716 | +3.3430 | +7.5330 | 0.00 | 0.0625 | not_supported |
| ema_vs_fourier | mae | 5 | +1.0053 | +0.6793 | +1.3396 | 0.00 | 0.0625 | not_supported |
| ema_vs_fourier | poisson_nll_no_const | 5 | +0.1005 | +0.0575 | +0.1355 | 0.00 | 0.0625 | not_supported |
