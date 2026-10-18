# V102: cross-day critic-only target repair

V101 completes 1120 arms without retry; its measured native soft gain is
+0.002448, 95% CI [-0.005136,+0.011870]. Passenger/CV/readiness intervals
also cross zero. Cached review `t179113` finds reward-curve split-half rank
correlation 0.861 versus critic/measurement 0.550, cost 0.752 versus 0.457.
Reward/cost action-difference MSE is 0.04851/0.01516 versus label-mean variance
0.00533/0.00111. Mean calibration is insufficient; conditional action-value
error may involve learning and state aliasing. Retain V99's separate objective
conflict, without interpreting V101 as an aggregate system-harm finding.

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

Frozen source `4255f2f592`. Smoke `t179138` on node004 passes 11 tests,
the full native timing/endpoint check, unchanged V101 forecast/effects and
both frozen-actor refits; its item is reused. Retrieved only the 61-byte marker.
Earlier smokes exposed an
inapplicable current-tree attribute and float32 gain arithmetic; both are
fixed, with no tolerance change. Earlier gated shards were cancelled before
launch. Current shards `t179139-t179141` cover [0,14), [14,17), [17,20),
with 3/2/2 workers soft-preferred to node001/002/004. Only the still-queued
first shard was reduced from seven workers when live capacity fell; its
cases, fit budget and marker are unchanged. Aggregate `t179144`
waits for their three markers. No repair effect is claimed before completion.
