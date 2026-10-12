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

Run: `native_transit_fit_transfer_stage198_diagnostic_20261013_r1`.
Next: use the mean8 training/held-out error contrast to choose between
optimization/representation changes and transfer-specific changes.
