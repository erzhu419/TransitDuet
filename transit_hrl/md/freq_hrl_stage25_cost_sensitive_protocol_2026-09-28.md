# Stage-25 Cost-Sensitive Causal Decisions

Freeze roots 209011/209061 and the same 16 states/root (two per path).
Preflight 208001 uses two states. No replay, new labels or controller updates.
Reuse Stage-15's 41 pre-intervention features including the two clocks;
neither counterfactual endpoints nor observed window outcomes enter inputs.
Hold out one whole path; standardize using training paths only, with constant
columns scaled by one. Fit 14 states/query two; preflight fits one/query one.

Training label y = observed window contrast + mean of the OLD eight Stage-18
tail draws. Candidate minimizes sum_i |y_i|/sum_j |y_j| times
log(1+exp(-sign(y_i)*f(x_i))) + ||theta||^2/2. Linear f includes an equally
penalized intercept. Zero labels have zero weight; all-zero labels give f=0.
Use L-BFGS-B, zero initialization, maxiter 256, gtol 1e-9, ftol 1e-12.
Choose now iff f>0; ties wait. No threshold or regularization search.

Matched controls: uniform-weight sign classification; MSE regression on
y/mean(abs(y)) with half mean squared loss and the same unit L2; short-window
cost-weighted classification trained on w alone (no tail correction).
All use the same features, folds and penalized intercept. Also retain fixed
always-now/always-wait. Four fits/fold, 32/root; preflight uses four old draws.

Score ONLY after fitting using w plus Stage-21's 64 cached tail draws (four
in preflight). Report (candidate_action-control_action)*(w+A) for all states,
path summaries, action/switch counts and conditional MC SE. A development
pass requires strictly positive mean benefit versus every control on BOTH
roots; ties fail. No new fits, root expansion or deployment follows a failure.
Use scheduleurm, dynamic node001-node006, 1 CPU/1536 MB per task, JSON only.
Nineteen focused tests cover gradient/cost algebra, causal input isolation,
whole-path fitting, scoring-label separation and scheduler input/resources.

## Limitations

These development states and scoring labels were previously inspected.
Fixed-state contrasts under the frozen continuation are not episode gains;
conditional MC SE excludes new-path uncertainty. Passing would qualify a
candidate for subsequent validation, not establish independent policy skill.
