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

## Limitations / Next Step

Results are conditional on four frozen actors from two original training seeds.
More fitting days may help prediction but do not establish action-effect
identification. Preserve old negative results and keep actor, dual and main
unchanged regardless of the outcome.
