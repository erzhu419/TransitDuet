# Stage141: Deployment-Mean Option Queries

Stage140 reduced exploration improves incremental returns in both modes but
does not repair mean deployment at period100. Sampled Fisher still trails
forecast at both periods. Test the deployment objective, not another std sweep.

Roots410037/410049, periods50/100, horizon1200, same Stage135 warm mean and strong
lower, std0.05, authority0.05. Reconstruct the Stage140 reduced-std stochastic
Fisher direction by exact path/credit replay. In the same eight training scenes
and two noise folds, execute a mean upper path; query each option at
mean+/-0.05*epsilon while every other option follows its conditional mean.
All prefixes, exogenous measurements and lower white-noise draws are paired.

The mean-query gradient target is(Rplus-Rminus)*epsilon/(2*std), passed through
the existing standardized damped mean geometry(damping1). Both branches use
mean-output RMS0.001767767/KL0.005, both update signs, and frozen lower/values.
No query trajectory is used as an on-policy PPO sample or critic target.
Evaluate32 fresh scenes/period under both mean and sampled upper; reuse only
the identical forecast row. Return and tracking reductions are reported for
each method versus warm/forecast and against the other method.

Per root32replay+32mean collection+1,152queries+704evaluation=1,920episodes,
2,304,000steps.576antithetic label pairs;64forecast aliases. Prior Stage140
costs remain separate. Two roots3,840episodes/4,608,000steps. Sixteen workers
plus parent/12GiB, dynamic node001-node006, no aggregator. Compact JSON only.

## Run Receipt

Implementation/results commit87336e2f82; preregistration commit1732eda848,
pushed before submission. Eleven related focused tests passed. Metadata reads
confirmed both Stage140 source cells and all four selected warm upper files.
Run: pointmaze_mean_option_query_stage141_probe_20261008_r1.
t136997/root410037 and t136998/root410049 were queued at receipt time, with
no assigned node. Preregistration and compact dispatch receipt are retained.

## Completed Result

t136997 finished on node004; t136998 on node006. Pulled246,409bytes of compact
JSON only. Exact contract, budgets, seeds, paired means and update RMS pass.
Total3,840episodes/4,608,000steps; no checkpoint or trace pulled.

| Period | Mean sampled-credit increment | Mean-query increment | Sampled mean-query increment |
| --- | ---: | ---: | ---: |
| 50 | +0.144882 | +0.210431 | +0.063844 |
| 100 | -0.034278 | -0.011610 | +0.038285 |

Mean-query improves on sampled-credit mean deployment at all four root-periods,
but root410049/period100 still loses0.115264 versus warm. Mean-query noise-fold
gradient cosines at100 are0.119442/0.036449, versus0.714641/0.306727 at50.
At100,96.35%/96.875% of paired option queries have negative even response;
mean even response is-0.248198/-0.258273. Sampled deployment still trails
forecast by0.431794/2.158360 at50/100. The mean-objective change helps but does
not close the period100 learning claim. Next: local versus wide query probes,
with policy std and mean-step RMS unchanged and source costs retained.

## Limitations

This compares a query-augmented smoothed mean-option gradient with an already
queried stochastic comparator, not equal new-query budgets. Finite probes and
mean-trajectory occupancy are different from stochastic PPO training. Step RMS
matches on each training batch, not decoded control distance. Two-root
development only; no confirmation CI, evaluation winner or joint HRL claim.
