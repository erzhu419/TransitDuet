# Stage243: variance-aware, group-shrunk service/risk critic

## Design

Reuse the Stage242 remote panel cache and Stage237/239 frozen raw references.
Service is nonfleet cost; risk is the fleet cost component, not wait-violation probability.
Fit both residual heads in the same198-feature TSVD space, with the original anchor
and the existing17 rank/strength candidates. Compare combined MSE, split OLS,
variance-weighted GLS, group shrinkage and their combination. Primary:
`variance_group_pooled_safe`; half-safe and the other arms are controls.

Estimate paired action-minus-original MC variance from training futures only.
Pool variance equally across prefixes, then half-pool group and state estimates.
Estimate each group's prior residual signal energy from the other training groups;
shrink its residual targets toward the frozen reference by signal/(signal+noise).
No prefix ID enters inference. Zero observed risk variance is reported as unidentified,
not certainty. Total residual RMS is constrained to the cached MSE budget by an exact
two-head convex GLS dual solve. All arms have k total-cost function degrees of freedom;
the split heads have2k component degrees of freedom, not extra total-cost capacity.

Select with unchanged inner prefix/future OOF nonharm rules. Compare old fit/unseen
labels and fresh4+4/8 on OOF prefixes, full training prefixes and outer held prefixes.
Split OLS must reproduce combined MSE at every rank/strength and selected action.
Seal models and actions before fresh-label evaluation. One real-cache cell precedes
expansion to32 cells; aggregate only after every cell finishes successfully.

## Limits And Next Step

Zero raw-reference fits, native episodes or actor updates. Small aggregates only
are pulled locally. Fresh futures are already-inspected development data, not new
confirmation or independent training-seed CI; Stage169 reserved cohorts stay closed.
Moment shrinkage and finite-draw precision weights are estimators, not risk guarantees.
Judge improvement by unseen-future MSE and held-prefix physical cost/wait together;
do not promote a model from lower training error alone. Defaults remain unchanged.
