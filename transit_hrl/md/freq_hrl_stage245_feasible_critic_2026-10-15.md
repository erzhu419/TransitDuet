# Stage245: feasible-action MSE and regret ranking

Fix Stage244 B-target/B-calibration, raw references,198 features, rank/strength
grid, wait gate and A-MSE residual RMS cap. Reuse cached all-action two-head GLS.
Compare equal-state feasible-action component MSE and gap-weighted feasible-pair
logistic regret ranking. Primary: `rank_regret_pooled_safe`; unchanged OOF rules.

Freeze the cached service-minus-risk residual direction; optimize only the cost
sum in the same k-dimensional TSVD space. Ranking targets are the same shrunk
physical service-plus-risk increments; new sum corrections are shared equally
between heads. The solver receives A's radius explicitly, not the smaller
shrunk initial norm. No root-specific policy parameters or new gate.

One real-cache cell precedes the32-cell matrix. Seal fits and actions before
fresh-future evaluation. Compare train/held prefixes, fresh4+4/8, root401 outcomes,
feasible MSE and gate regret. Shared outcomes count once in conditional MC SE.
No raw-reference training, new native replay or extra target/calibration draws.

Next: judge loss alignment by held-prefix cost/wait, not training objective alone.
Limits: reused development data, not confirmation or training-seed CI. Ranking
component allocations retain a frozen contrast, not new independent head fits.
Defaults and Stage169 reserves remain unchanged; full models stay remote.
