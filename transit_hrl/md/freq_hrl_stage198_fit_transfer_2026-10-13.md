# Stage198: fitting versus transfer

Reuse all188 Stage197 models,600 causal inputs and unchanged mean8/singleton
targets. No fitting, actor updates or native collection. One1CPU/3072MB
scheduler task, unbound node001-node006; raw artifacts remain server-only.

Compare physical pair MSE with a zero-gap reference separately on mean8
training, singleton training and held-out mean8 states. Training errors are
unadjusted for label noise. Report final fitted-objective loss separately;
held-out predictions come from the exact Stage197 archive. Repeated training
evaluations across folds are not new states or independent seeds.

Check lower-root occupancy/plan feature shifts in training input scales.
Root-policy and demand distributions are confounded, so these shifts alone
cannot establish a causal lower-policy effect. Tests:45 passed.

Run: `native_transit_fit_transfer_stage198_diagnostic_20261013_r1`;
`t147385`, node006, done/exit0. Inference/summary10.09s,206182-byte report
pulled once; no model or raw-data pulls.

Weighted advantage, lower-root: cost MSE/zero falls to0.635 on mean8 training
but rises to1.141 held-out; wait0.246 versus1.103. Uniform advantage fits
singleton costs (0.679) better than mean8 costs (1.027). All four models'
training objectives decreased. Checked26 physical/plan feature mean shifts
are modest (max absolute0.277 training standard deviations).

Next: test a lower-capacity pair-ridge predictor on identical features,
targets and folds before expanding the nonlinear model or collecting seeds.
Limits: in-bag errors are optimistic; modest marginal shifts do not exclude
conditional/covariance shifts, and root-policy/demand effects are confounded.
