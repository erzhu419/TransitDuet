# Stage199: lower-capacity action-credit baseline

After Stage198's in-bag/held-out gap, compare fixed ridge0.1 against the cached
neural advantage models, each with uniform versus count weights. Identical
98 causal features, training input/target scales,600 inputs,240 mean8 targets
and47 grouped folds. Ridge has294 coefficients and a closed-form solve;
neural controls have1635 parameters and100 updates. Capacity and optimization
are deliberately different; this is not a matched-budget superiority claim.

Primary: ridge/count-weighted, wait-feasible. Unchanged negative-cost/nonpositive-
wait gate on both roots across all three splits. No tuning or reserved scenes.
94 new solves,94 cached neural models,0 optimizer/actor/native updates.
Tests:23 passed covering ridge, weighted credit and fit-transfer diagnostics.

Run: `native_transit_pair_ridge_stage199_development_20261013_r1`.
Scheduler:47 fits then one report,1CPU/3072MB, unbound node001-node006.
Completed: `t147387-t147433` and report `t147449`, all done/exit0.
One104126-byte report pull; no model or raw-data pulls.

Weighted ridge, wait-feasible, pooled cost / wait-minute deltas:

| Split | Cost | Wait | Cost MSE / Zero Gap MSE |
| --- | --- | --- | --- |
| Scene | -0.001641 | -0.001317 | 0.889 |
| Regime | -0.001709 | -0.000168 | 0.950 |
| Lower root | -0.002809 | -0.001293 | 0.883 |

Both ridge variants pass the cost gate on both roots across all three splits.
Weighted ridge fails only scene/root397 wait: +0.001229 minutes; its cost is
-0.002181 there. The strict combined gate remains closed.

Next: retain the ridge cost predictor and separately calibrate the wait
constraint using grouped training-only predictions. No held-out margin tuning
or new seeds. Limits: these are development point estimates, not CI-supported
improvements or full upper/lower HRL validation; optimization/capacity differ.
