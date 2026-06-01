# Public GTFS Demand-Proxy Validation

- source URL: `https://www.bart.gov/dev/schedules/google_transit.zip`
- cached feed: `transit_hrl/data/public_gtfs_bart/google_transit.zip`
- bin size: `1800s`
- data path: public GTFS `stop_times.txt`, converted into causal stop-level scheduled event bins
- boundary: scheduled events are a real Transit activity proxy, not AFC/APC passenger counts
- best by MSE: `dynamic_harmonic_nb`

| method | series | MSE | MAE | Poisson NLL | delta MSE vs best |
|---|---:|---:|---:|---:|---:|
| dynamic_harmonic_nb | 24 | 803.8486 | 23.7887 | 66.3498 | +0.0000 |
| ema | 24 | 947.1398 | 26.7743 | 1.0075 | +143.2912 |
| fourier | 24 | 963.2267 | 26.9218 | 24.5309 | +159.3781 |

## Paired Method Deltas

Deltas are `method - fourier`; lower is better for all listed metrics.

| comparison | metric | n | delta | CI95 low | CI95 high | win rate | status |
|---|---|---:|---:|---:|---:|---:|---|
| dynamic_harmonic_nb_vs_fourier | mse | 24 | -159.3781 | -226.8801 | -96.9405 | 0.75 | supported |
| dynamic_harmonic_nb_vs_fourier | mae | 24 | -3.1331 | -4.4426 | -1.8983 | 0.79 | supported |
| dynamic_harmonic_nb_vs_fourier | poisson_nll_no_const | 24 | +41.8189 | +26.4071 | +58.3771 | 0.04 | not_supported |
| ema_vs_fourier | mse | 24 | -16.0869 | -17.7398 | -14.1868 | 1.00 | supported |
| ema_vs_fourier | mae | 24 | -0.1475 | -0.1626 | -0.1333 | 1.00 | supported |
| ema_vs_fourier | poisson_nll_no_const | 24 | -23.5234 | -26.1206 | -20.8571 | 1.00 | supported |
