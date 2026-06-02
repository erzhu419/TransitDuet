# Public AFC Demand Validation

- source: `https://data.ny.gov/resource/wujg-7c2s.json`
- window: `2024-10-01T00:00:00` to `2024-10-15T00:00:00`
- rows fetched: 140078
- data path: NY Open Data/MTA hourly subway ridership, aggregated by station complex and hour
- boundary: subway station entries are AFC-style passenger demand; they are not APC onboard loads or OD flows
- best by MSE: `ema`

| method | series | MSE | MAE | Poisson NLL | delta MSE vs best |
|---|---:|---:|---:|---:|---:|
| ema | 24 | 7606159.9467 | 1866.6540 | -7159.3266 | +0.0000 |
| fourier | 24 | 7671139.1898 | 1865.7857 | -6661.3085 | +64979.2431 |
| dynamic_harmonic_nb | 24 | 7687260.9245 | 1869.6716 | -6492.5549 | +81100.9778 |

## Paired Method Deltas

Deltas are `method - fourier`; lower is better for all listed metrics.

| comparison | metric | n | delta | CI95 low | CI95 high | win rate | status |
|---|---|---:|---:|---:|---:|---:|---|
| dynamic_harmonic_nb_vs_fourier | mse | 24 | +16121.7347 | +6046.0604 | +31800.8527 | 0.08 | not_supported |
| dynamic_harmonic_nb_vs_fourier | mae | 24 | +3.8859 | +2.7059 | +5.5217 | 0.00 | not_supported |
| dynamic_harmonic_nb_vs_fourier | poisson_nll_no_const | 24 | +168.7536 | +98.8277 | +265.4855 | 0.12 | not_supported |
| ema_vs_fourier | mse | 24 | -64979.2431 | -89698.3451 | -44124.1167 | 1.00 | supported |
| ema_vs_fourier | mae | 24 | +0.8683 | +0.0645 | +1.4578 | 0.25 | not_supported |
| ema_vs_fourier | poisson_nll_no_const | 24 | -498.0181 | -578.1577 | -419.3571 | 1.00 | supported |
