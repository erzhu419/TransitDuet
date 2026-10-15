# V64: independent training days x value representation

Four V59 post-64 actors and their upper/B200 deployment stay frozen. Collect
32 independent fitting demand days (640001-640032) and eight fresh validation
days (740001-740008) per actor: 160 full-day rollouts. Every actor/day has a
distinct private policy sampler seed, `6400000 + case_index * 1000 + day_index`.
Neither dataset reuses V62/V63 validation seeds. State arrays stay on server.

The small fitting set is fixed as the first four of the 32 training days.
At each size, fit direct signed remaining CV-squared returns and terminal
CV-squared prediction minus the exact completed prefix. Both representations
start from identical V59 cost weights, use the original architecture/LR,
2,048 Adam steps and batch 256. Pair minibatch draws using the same seed;
keep the total update budget equal across data sizes. Sixteen fitted models.
Save both representations under distinct semantic keys before evaluation.

Evaluate all models on the same eight full held-out trajectories per actor.
Report zero, size-specific causal final-mean priors, V62 direct and V59 frozen
controls. Large-model strict gate: both MAE/RMSE beat every control and both
small models on every validation day, for all four actors. Calibration can
admit a later action audit, never control changes in this experiment.

For data-size and representation contrasts, average actor-context paired
deltas within each demand seed, then bootstrap eight day clusters (10,000
resamples, fixed seed). Report mean/CI and day deltas; no decision-level or
independent-training-seed inference. Tests and a full collection smoke must
pass before the CPU batch; fitting/evaluation/aggregation use scheduler on
node001-node006 only. Pull completion markers and compact summary JSON only.

## Outcome

Server tests: 26 passed. Completed 160 rollouts (832,000 decisions), 16 fits
and 32 held-out actor/day evaluations (166,400 decisions).

| Model | 4-day MAE / RMSE | 32-day MAE / RMSE |
| --- | --- | --- |
| Direct remaining return | 0.009933 / 0.011816 | 0.009538 / 0.011462 |
| Terminal value minus prefix | 0.011584 / 0.012128 | 0.010259 / 0.011501 |
| Fitting-day mean minus prefix | 0.010294 / 0.010294 | 0.009571 / 0.009571 |

Day-cluster paired MAE deltas (32 minus 4 days): direct -0.000395,
95% CI [-0.001541, 0.000817]; factored -0.001324 [-0.005452, 0.003130].
At 32 days, factored minus direct: +0.000721 [-0.004723, 0.005974].
Both strict gates: 0/4 contexts; no action audit or control admission.
Actor/dual updates: zero. Main unchanged. Data scarcity alone is not established
as the explanation. Training losses declined but do not establish convergence.

Scheduler records: t165310-t165327. Placement-only change: t165312 (18 workers)
was cancelled and replaced by t165327 (10 workers) when CPU availability fell;
its demand days, seeds and output range stayed unchanged.
Small summary: `results_freqduet/protocol_v6_v64_independent_days/summary/status.json`.

## Limitations / Next Step

Results are conditional on four frozen actors from two original training seeds.
Intervals resample validation days, not fitting sets or the fixed four-day subset.
More fitting days may help prediction but do not establish action-effect
identification. Preserve old negative results and keep actor, dual and main
unchanged regardless of the outcome.
Next compare a low-capacity residual model anchored to the causal fitting-day
prior against the same-budget neural models, using only V64 training days for
development and reserving new demand days for confirmation. Do not increase
actor training or tune a gate on these eight now-consumed validation days.
