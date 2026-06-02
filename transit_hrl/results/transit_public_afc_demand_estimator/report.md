# Public AFC Demand Validation

- source: `https://data.ny.gov/resource/wujg-7c2s.json`
- window: `2024-10-01T00:00:00` to `2024-10-15T00:00:00`
- rows fetched: 140078
- data path: NY Open Data/MTA hourly subway ridership, aggregated by station complex and hour
- boundary: subway station entries are AFC-style passenger demand; they are not APC onboard loads or OD flows
- best by MSE: `afc_daily_profile`

| method | series | MSE | MAE | Poisson NLL | delta MSE vs best |
|---|---:|---:|---:|---:|---:|
| afc_daily_profile | 24 | 484904.0070 | 378.7106 | -13273.8077 | +0.0000 |
| ema | 24 | 7606159.9467 | 1866.6540 | -7159.3266 | +7121255.9397 |
| fourier | 24 | 7671139.1898 | 1865.7857 | -6661.3085 | +7186235.1828 |
| dynamic_harmonic_nb | 24 | 7687260.9245 | 1869.6716 | -6492.5549 | +7202356.9175 |

## Paired Method Deltas

Deltas are `method - fourier`; lower is better for all listed metrics.

| comparison | metric | n | delta | CI95 low | CI95 high | win rate | status |
|---|---|---:|---:|---:|---:|---:|---|
| afc_daily_profile_vs_fourier | mse | 24 | -7186235.1828 | -11227317.0177 | -4426375.2393 | 1.00 | supported |
| afc_daily_profile_vs_fourier | mae | 24 | -1487.0751 | -1861.9975 | -1211.7629 | 1.00 | supported |
| afc_daily_profile_vs_fourier | poisson_nll_no_const | 24 | -6612.4993 | -8186.3867 | -5459.8890 | 1.00 | supported |
| dynamic_harmonic_nb_vs_fourier | mse | 24 | +16121.7347 | +6049.5629 | +30336.5828 | 0.08 | not_supported |
| dynamic_harmonic_nb_vs_fourier | mae | 24 | +3.8859 | +2.7104 | +5.3666 | 0.00 | not_supported |
| dynamic_harmonic_nb_vs_fourier | poisson_nll_no_const | 24 | +168.7536 | +98.3379 | +252.4535 | 0.12 | not_supported |
| ema_vs_fourier | mse | 24 | -64979.2431 | -91793.5861 | -44541.0252 | 1.00 | supported |
| ema_vs_fourier | mae | 24 | +0.8683 | +0.1866 | +1.4558 | 0.25 | not_supported |
| ema_vs_fourier | poisson_nll_no_const | 24 | -498.0181 | -584.3150 | -422.2208 | 1.00 | supported |
