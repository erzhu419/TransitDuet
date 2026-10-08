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

## Limitations

This compares a query-augmented smoothed mean-option gradient with an already
queried stochastic comparator, not equal new-query budgets. Finite probes and
mean-trajectory occupancy are different from stochastic PPO training. Step RMS
matches on each training batch, not decoded control distance. Two-root
development only; no confirmation CI, evaluation winner or joint HRL claim.
