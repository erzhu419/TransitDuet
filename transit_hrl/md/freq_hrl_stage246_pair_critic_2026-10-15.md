# Stage246: variance-weighted paired-margin regression

Freeze Stage245 raw/all-MSE/feasible-MSE/rank-regret models and selected controls.
Compare equal-state feasible pair-MSE with pair-GLS, using the same shrunk B
service/risk targets,198 features, ranks1/3/8/16,17 candidates and explicit A RMS
cap. Primary: `pair_gls_pooled_safe`; unchanged prefix/future OOF selection/gate.
Cached head difference remains fixed; only the k-dimensional cost sum is learned.

Estimate each head's variance from same-future action differences, retaining
common-shock covariance. Pool all15 pairs within prefix groups and globally using
the existing half/half rule; normalize precision per head. The loss uses only
feasible pairs. Zero observed risk variance means unit precision, not known-zero risk.

One real-cache preflight precedes the32-cell matrix and report. Seal models/actions
before fresh-future evaluation. Compare held/train-prefix cost/wait, feasible
margin MSE, pair-sign error, changed-harm rate and conditional fleet-event sampling.
No raw-reference refit, native replay, new target draws or reserve access.

Next: distinguish pair geometry from noise weighting using pair-MSE versus pair-GLS;
judge actual cost/wait rather than training loss. Reused development futures are
not confirmation or training-seed CI. Defaults unchanged; full models stay remote.
