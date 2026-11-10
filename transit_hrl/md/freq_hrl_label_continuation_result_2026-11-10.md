# Paired Label Continuation Result

All eight probes t214367-t214374 exited 0: 32 fresh paths, 96 sampled states,
eight futures/state and 4,185,600 native steps, exactly the frozen budget.
Probe cost: 3,556.946 CPU seconds; no fits or controller updates. Retrieved
705,522 public-result bytes including a cache-only score audit, no models/raw banks.

## Primary Comparison

C-label head minus R-label head on the SAME fixed-C0 prefixes and continuation.
MSE changes are worse when positive; W/G action gains are better when positive.

| Task / root | Fresh W MSE change | W gain | G gain | Changed actions |
| --- | ---: | ---: | ---: | ---: |
| PointMaze / 1972110000 | +0.85% | +0.000459 | +0.000607 | 1/24 |
| PointMaze / 1972120000 | -0.31% | 0 | 0 | 0/24 |
| Reacher / 1972210000 | +10.10% | -0.000899 | -0.010402 | 12/24 |
| Reacher / 1972220000 | +8.60% | +0.000504 | +0.008559 | 6/24 |

Both models are scored against the common C0 target, including training A.
Training-A error falls on all four roots; independent-B error improves on three.
Reacher root1 goes -41.28% on A, -42.15% on B, -4.94% on prefix-OOF-B, then
+10.10% on new prefixes (+11.46%/+8.92% across future halves). Relabeling helps
known-prefix prediction, but its improvement does not reliably transfer.

PointMaze's incremental gain comes only from path1972160007/step450; root2
has no action changes. Reacher root1 genuinely changes half its sampled actions,
yet incremental W/G changes are negative in BOTH future halves. It is not a disconnected
head or a universal score-offset issue. C-label versus frozen C0 itself has
negative W AND G point gains on three roots, including Reacher root2: improvement
over the R-label refit there is not an absolute improvement over C0.

## Decision And Next

The continuation-only repair did not meet the declared stable-improvement goal.
No closed-loop follow-up or new experiment is submitted; HOLD remains.
Next isolate decision calibration on frozen scores: training-A prefix-OOF
paired decision utility versus the fixed score quantile, with B kept independent.
Do not infer a repair by changing cuts on these revealed evaluation paths.

Limits: two roots/task; descriptive unadjusted path CIs, with no strictly positive
root-level incremental W/G gain interval. OOF conditions on frozen C0, not the
whole pipeline. These are single interventions, not new-head closed-loop returns.
Earlier lifecycle-cost and termination-competence gaps remain unresolved.

Sources: [summary](../results/plan_validity_label_continuation_20261110/summary.json),
[audit](../results/plan_validity_label_continuation_20261110/execution_audit.json),
[frozen cache readout](../results/plan_validity_label_continuation_20261110/frozen_cache_readout/public/result.json).
