# Transit Demand Estimator Validation

- best by MSE: `fourier`

| method | seeds | MSE | MAE | Poisson NLL | delta MSE vs best |
|---|---:|---:|---:|---:|---:|
| fourier | 5 | 9.2595 | 2.3179 | -39.6470 | +0.0000 |
| dynamic_harmonic_nb | 5 | 12.8485 | 2.7869 | -39.6087 | +3.5890 |
| ema | 5 | 14.6311 | 3.3232 | -39.5465 | +5.3716 |
