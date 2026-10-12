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
Next: assess whether regularized action differences retain held-out signal and
actual cost/wait benefits. Models, bank, targets and predictions stay remote.
