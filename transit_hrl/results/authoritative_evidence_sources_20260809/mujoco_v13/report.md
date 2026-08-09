# MuJoCo v13 behavioral Confirmatory Decision

- status: `confirmatory_primary_gate_failed`
- integrity: `valid`
- primary family-wise gate: `False`
- optimizer replicates per environment and arm: `24`
- held-out paths per cell: `40`
- family-wise alpha: `0.05` across `12` statistical gates

| Environment | Return delta [95% CI] | NI | Responsibility drift reduction | Raw lower drift reduction | Upper HF RMS |
| --- | ---: | --- | ---: | ---: | ---: |
| HalfCheetah-v5 | 96.059 [32.206, 176.069] | True | 74.89% (True) | 5.22% (False) | 0.0634 (True) |
| Hopper-v5 | 13.025 [3.820, 24.158] | True | 73.15% (True) | 58.04% (True) | 0.1068 (False) |
| Walker2d-v5 | 23.406 [11.618, 36.420] | True | 90.37% (True) | 38.39% (True) | 0.0240 (True) |

Return superiority is exploratory; the preregistered return claim is noninferiority.

Development disclosure: The behavioral endpoints and numerical thresholds were developed after exploratory inspection of MuJoCo v12. All optimizer, training, checkpoint, safety-selection, and held-out evaluation seeds are fresh for v13, and the v13 decision rule is frozen before access to any v13 held-out result.
