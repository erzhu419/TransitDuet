# V102: cross-day critic-only target repair

V101's 1120 arms complete without retry. The critic-implied distribution has
predicted soft gain +0.004407; measured native soft gain is +0.002448,
95% CI [-0.005136,+0.011870]. With the original uncertainty charge retained,
gain is +0.004722 [-0.002715,+0.015691]. Reward and cost contributions are
+0.002476 and -0.000242; both intervals cross zero. No stable passenger,
CV-squared or readiness benefit/harm is established for this one-action change.

Cached review `t179113` passes three tests, performs no rollout/update, and
returns only 136456 bytes. Reward-curve split-half rank correlation is 0.861
versus critic/measurement 0.550; cost is 0.752 versus 0.457. Action-difference
MSE is 0.04851/0.01516 versus label-mean variance 0.00533/0.00111. There is
conditional action-value error beyond observed continuation noise, despite
small mean gain-prediction error. Of 40 contexts, 22 have positive measured
fixed-charge soft gain in both halves; among those, 11 have higher passenger
time, 10 higher CV squared and 10 higher readiness delay by point estimate.
These counts are descriptive, not simultaneous discoveries or a policy gate.

Test a critic-only repair before any actor distillation. Replay the same 20
native B200 paths to recover exact trip chains. For each actor, fit on one
consumed day and evaluate only the other day's V101 states and all seven
cached action branches; then reverse the days. Do not fit counterfactual labels.
Compare untouched critics, 64 original TD updates and 64 native Monte Carlo
regression updates, batch 512 with identical sample indices and optimizer
initialization. Keep architecture, regularization, gamma, alpha and dual fixed.
MC labels include future entropy but not current entropy. Reuse the native
critic updater with terminal regression labels; require labels within its
existing bounds so no MC clipping occurs. No new coefficient or step search.

Primary repair evidence is paired held-day action-difference MSE for reward,
cost and combined score: MC versus both TD and untouched. Also report MAE,
noise-adjusted squared error and the same one-decision expected local/physical
effects. Hold the original uncertainty charge fixed across forecast models.
Accuracy improvement alone cannot repair V99's system-objective conflict or
authorize deployment. Do not reinterpret a critic refit as a daily controller.

Smoke must pass focused batch/target/freeze tests, reproduce the complete
native timing path, and reproduce V101's untouched forecast/effects before
releasing the 20-item matrix. Reuse its completed item. All work uses scheduler
on node001-node006 and the existing isolated CPU environment. Pull only markers
and compact summaries. Main, actor, upper and dual remain unchanged; copied
critics update only offline and never control the simulator.

Limitations: development on two consumed days; cross-day testing is not new-day
confirmation. Four futures and 40 visited prefixes cannot establish general
value accuracy. A null result ends this fixed-budget candidate, not all MC
methods. A positive result requires fresh-day confirmation and separate system
credit repair before considering a policy change.
