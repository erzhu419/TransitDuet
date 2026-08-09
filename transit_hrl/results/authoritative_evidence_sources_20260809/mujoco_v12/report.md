# MuJoCo v12 Confirmatory Decision

- status: `confirmatory_supported`
- integrity: `valid`
- primary family-wise gate: `True`
- optimizer replicates per environment and arm: `24`
- held-out paths per cell: `40`
- family-wise alpha: `0.05` across `6` statistical gates

| Environment | Return delta [95% CI] | NI gate | Drift reduction [95% CI] | Drift gate |
| --- | ---: | --- | ---: | --- |
| HalfCheetah-v5 | 118.704 [53.010, 198.837] | True | 89.60% [79.10%, 94.26%] | True |
| Hopper-v5 | 20.241 [5.193, 38.992] | True | 78.23% [69.90%, 85.28%] | True |
| Walker2d-v5 | 31.402 [17.650, 46.177] | True | 89.56% [85.13%, 93.55%] | True |

Return superiority is exploratory; the preregistered return claim is noninferiority.
