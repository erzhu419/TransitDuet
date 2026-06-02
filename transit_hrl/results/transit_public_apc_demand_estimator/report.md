# Public APC Demand Validation

- source: `https://services2.arcgis.com/11XBiaBYA9Ep0yNJ/ArcGIS/rest/services/Transit_Automated_Passenger_Counts/FeatureServer/0/query`
- window: `2026-01-01` to `2026-02-01`
- rows fetched: 50000
- data path: Halifax Transit half-hourly route boardings collected by bus Automatic Passenger Counters
- boundary: route boardings are real APC passenger demand; they are not onboard occupancy, alightings, or OD flows
- best by MSE: `apc_route_profile`

| method | series | MSE | MAE | Poisson NLL | delta MSE vs best |
|---|---:|---:|---:|---:|---:|
| apc_route_profile | 24 | 1790.6664 | 15.5078 | -161.1020 | +0.0000 |
| dynamic_harmonic_nb | 24 | 6447.9648 | 44.8465 | 26.8900 | +4657.2984 |
| ema | 24 | 6707.0844 | 45.2752 | -40.6189 | +4916.4179 |
| fourier | 24 | 6825.6658 | 45.8214 | -24.6012 | +5034.9993 |

## Paired Method Deltas

Deltas are `method - fourier`; lower is better for all listed metrics.

| comparison | metric | n | delta | CI95 low | CI95 high | win rate | status |
|---|---|---:|---:|---:|---:|---:|---|
| apc_route_profile_vs_fourier | mse | 24 | -5034.9993 | -8716.7818 | -2291.3921 | 0.96 | supported |
| apc_route_profile_vs_fourier | mae | 24 | -30.3136 | -41.1008 | -20.8317 | 1.00 | supported |
| apc_route_profile_vs_fourier | poisson_nll_no_const | 24 | -136.5008 | -177.9653 | -100.1271 | 1.00 | supported |
| dynamic_harmonic_nb_vs_fourier | mse | 24 | -377.7009 | -756.7249 | -71.1664 | 0.71 | supported |
| dynamic_harmonic_nb_vs_fourier | mae | 24 | -0.9749 | -2.5126 | +0.5905 | 0.75 | positive_mixed |
| dynamic_harmonic_nb_vs_fourier | poisson_nll_no_const | 24 | +51.4912 | +19.4515 | +93.1566 | 0.12 | not_supported |
| ema_vs_fourier | mse | 24 | -118.5814 | -201.3133 | -56.8909 | 1.00 | supported |
| ema_vs_fourier | mae | 24 | -0.5462 | -0.7215 | -0.4049 | 1.00 | supported |
| ema_vs_fourier | poisson_nll_no_const | 24 | -16.0177 | -21.3527 | -11.6392 | 1.00 | supported |
