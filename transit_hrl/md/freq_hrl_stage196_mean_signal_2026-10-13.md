# Stage196: Conditional Noise Versus Critic Error

Reuse Stage195's240 states,40 development prefixes, eight conditional draws and
47 frozen fits. For each cost/fleet/nonfleet/wait head, subtract the mean-label
pair variance from observed pair MSE without clipping negative estimates.
Compare against zero action differences using the same noise subtraction.
This is conditional Monte Carlo error analysis, not a cross-seed confidence CI.

Check oracle choice transfer between odd/even four-draw halves. Split selected
false wait-feasibility into harm in both halves versus half disagreement.
Oracles are diagnostics, not deployable policies or causal performance claims.

One scheduler report task, nodes001-006 unbound. No new fits, actor updates or
native episodes. Raw draws/predictions remain remote; only summary is synced.

## Result
Task`t147224` completed exit0, source`63b0634b82`; seven tests pass. One188,343
byte summary was pulled. Half choices agree98/240. Hindsight cost-0.008184
shrinks to cross-half cost-0.001150/wait-0.015729min; both roots remain negative
in cross-half means. This is a sampled opportunity diagnostic, not a policy win.

Mean-target cost pair MSE after noise subtraction is0.000573/0.000734/0.000779
for scene/regime/lower-root, versus zero-reference0.000452. Fleet accounts for
most of the error excess. Wait is only slightly better than zero on scene and
worse on the other splits;49/76,51/78,44/67 false feasible selections harm wait
in both halves. The failed screen is not explained by label noise alone.

Read-only server inspection of the existing training ledgers/predictors finds
normalized pair loss20-30% below zero-reference, but physical cost pair MSE only
4-5% lower. Thus cost credit is weak even in training, with worse held-out error.
These empirical training losses are not noise-corrected population estimates.

## Next Step
Keep the cohort/gate fixed. Test mean-target reliability weighting separately
from learning direct action advantages:360 single labels still dominate much
of the training noise, while absolute fleet anchoring poorly learns cost gaps.
Refit the wait constraint with the same critic changes; no extra seeds or
reserved-scene release until grouped cost/wait and difference-error tests pass.
