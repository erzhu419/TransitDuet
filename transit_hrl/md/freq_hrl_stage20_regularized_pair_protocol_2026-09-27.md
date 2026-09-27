# Stage-20 Regularized Paired Predictor

27 focused tests passed: analytic ridge solution, fixed regularization under
row replication, pair symmetries, path isolation and existing critic behavior.

Frozen roots: 209011/209061; operational preflight: 208001. Reuse Stage-18's
16 opportunities/root and eight futures/opportunity, Stage-16 endpoints and
Stage-19 out-of-path neural predictions. No new environment samples or roots.

Replace the neural value function with a shared linear value difference.
For compact endpoint x, phi(x) = remaining_fraction * (x - train_mean)/train_sd;
D = phi(wait) - phi(now). Use all 40 features, no intercept, interactions or
clipping; omit the 384 all-zero slots. Endpoint normalization and path weights
match Stage-19. This preserves antisymmetry, identical-state zero and terminal
zero. Fit argmin_beta sum_i w_i (D_i beta - label_i)^2 + ||beta||^2, with
weights summing to one and equal mass per path. Unit L2 is fixed before fits,
without a grid or held-out selection. Solve the 40-dimensional ridge system.

Two label treatments: replica 0 and eight-replica mean. Each whole-path fold
uses 14 training pairs and two queries; all replicas of the query path stay
outside fitting/normalization. The mean-label ridge is the sole candidate;
single-label ridge is a diagnostic, not an alternative chosen after results.
It must beat zero AND Stage-19 mean-label neural MSE on BOTH roots. Report
single-label controls, path metrics, coefficients and effective degrees of
freedom. Use the same raw/repeat-variance-corrected MSE as Stage-19; no clipping
of negative finite-sample estimates. No alpha, feature or root changes after fit.

Sixteen closed-form fits/root, zero gradient steps and zero controller replay.
Preflight: two paths, one pair/path, four futures and four closed-form fits.
Inherited Stage-18 cost remains 547,200 replay steps/root plus earlier training.
Schedule through node001-node006 without pins; sync compact JSON only.

This jointly tests lower capacity and regularization, not their separate causal
effects. Reused development states cannot confirm a deployment claim; even a
pass requires a separate control-utility protocol. Preserve Stage-19 failures.

Implementation `76fca5c01b`. Preflight `t101365` completed on node004;
23,821-byte JSON, four 40-coefficient fits, eight cached labels, no gradient
updates or new environment steps. Pairing, path isolation, matched normalization
and metric recomputation passed. Corrected MSE: ridge mean 2.820614e-7,
ridge single 1.075261e-5, zero/Stage-19 mean -4.954739e-8. Qualification failed;
no parameter changes precede the development run.
