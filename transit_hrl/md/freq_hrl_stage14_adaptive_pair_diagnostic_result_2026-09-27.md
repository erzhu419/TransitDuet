# Stage-14 Adaptive-Continuation Result

Tasks `t101002/101003` completed. The 64/61 registered opportunities,
230,400/219,600 replay steps, exact three-arm prefixes, factual decision
times/ISE/return, and 24-call budgets passed. No outcome was used for fitting.

| Root | Factual action | N | Chosen-action full ISE benefit versus one-check alternative | Positive benefit | Now versus deadline full ISE benefit |
|---|---|---:|---:|---:|---:|
| 209011 | Now | 32 | 0.004277 | 24/32 | 0.066116 |
| 209011 | Wait | 32 | 0.054897 | 22/32 | -0.054897 |
| 209061 | Now | 29 | 0.020107 | 25/29 | 0.061322 |
| 209061 | Wait | 32 | 0.019119 | 18/32 | -0.019119 |

Actual early calls often have a useful but much smaller benefit over waiting
one check than over forcing the deadline. Among early-call samples, the two
contrasts disagree in full-episode sign on 6/32 and 4/29 opportunities;
old score versus one-check full benefit correlations are -0.005/-0.280.
This is not a blanket score failure: both factual strata retain positive
mean benefit on both roots. The Stage-12 performance gate remains failed.

Next: fit one-check, full-episode advantage on branch-fit paths only, with
adaptive reference-policy continuation and explicit budget/episode clocks.
Use zero advantage for the new action comparison, not a retuned quantile of
the old predictor. These revealed evaluation paths remain development data.
