# Stage267: Coupled Full/Prefix Noise

Same caches: d_delta = -T_prefix d_g_prefix + T_full d_g_full.
Retain shared-label covariance; also verify one_ended + operator_change.
t165743/node006: exit 0, 73.96s; 20 tests passed; 32 cells / 1,280 records.
Primitive Stage266 replay error <= 5.06e-14. Pulled 2.59 MB scalar report;
raw ledgers, matrices and full-band aggregates remain remote.

Mixed removed held queries, free_old / shrunk: relative to Stage266 one-ended
noise, service variance falls 91.7% / 84.1%, joint variance 62.7% / 50.4%.
Joint variance is only 13.6% / 14.0% of independent endpoint variance sums.
Joint observed/noise RMS is .98 / .92; service 1.15 / 1.41.
Focal service-noise share falls from 89.1% / 79.8% to 24.3% / 25.3%.
Focal risk variance instead rises 65.2% / 61.2% versus the one-ended diagnostic.
Active 16@1 noise-norm/radius falls from 4.32 / 2.94 to 1.00 / .77.

Numerical repair: r1/t165732 failed a derived-ratio check at roundoff-zero
one-dimensional active tangents. Exact zero tangents and primitive replay
fixed r2; the original registration/failure remain recorded, thresholds unchanged.

Limitations: frozen-calibration local two-bank noise, not finite-refit error,
performance or CI. Response sizes are not uniformly small; signed covariance
cannot be treated as causal shares. Stage253/248/169 remain closed; no new fits,
simulations, selection or default changes in either attempt.

Next: Stage268 should decompose cached paired fleet-risk events and per-future
influence in the coupled response, including population/operator effects.
Do not infer an across-the-board shrinkage remedy from one-ended noise.
