# Frozen Diagnostic Result

t214162-t214165 all completed. All12 saved selection means reproduced exactly;
all same-prefix/factual-replay checks passed. No controller, ridge or termination
updates occurred. Cost:2,044.031 CPU seconds,2,485,200 simulator steps; only
62,546 bytes of public outputs were retrieved. Parent decision remains HOLD.

## Complete Episodes

Reserved development evaluation paths:16/root,2 roots/task. Mean ISE changes
relative to comparators; negative is better:

| Task | Versus Threshold | Versus Termination |
| --- | ---: | ---: |
| PointMaze | -4.83% | -14.28% |
| Reacher | +0.023% | -0.93% |

PointMaze wins both root means, but the pooled second8-path half is slightly
worse than threshold (+0.00505 ISE). Reacher's threshold contrast nearly cancels: root1972210000
loses to both comparators in both episode halves;1972220000 wins both in both
halves. The failed termination eligibility is retained, not repaired by this test.

## Three Causes

**Prediction contributes, but is not sufficient.** Reacher deployed cross-block
MSE/zero-predictor MSE is0.78/3.04 across the two roots. Replacing scores with
conditional local means at the SAME frozen threshold improves sampled full-episode
effects in both future directions for both roots (mean gains0.00302/0.01818).
Yet the root with worse local MSE wins the complete-episode comparison.

**Short credit misses important continuation effects.** Reacher adaptive50-step
advantages average0.00163/0.00539, but episode advantages are+0.03510/-0.06765;
tail contributions are+0.03347/-0.07304. Local/full signs disagree in25-58% of
cross-half probe comparisons. Local half-sign agreement is92-100%, versus67-83%
for full effects. PointMaze also loses most positive no-more-calls local benefit
to the opportunity/tail terms. Simply predicting the old50-step target better
does not establish better episode decisions.

**Absence of opportunity is not established.** Full-signal information actions
beat the frozen action in both future directions for both Reacher roots. Their
extra gain over block-selected constant now/wait actions is much smaller
(0.00342/0.00150); the second root's directions disagree. More complicated
state-dependent timing is not yet shown to have a large, reliably learnable gain.
The score>0 intervention benchmark is worse for both PointMaze roots in both
future directions, and mixed on the second Reacher root. A universal threshold fix
is therefore unsupported.

## Next Step And Limits

The next repair study should compare L/W/G targets on identical TRAINING-role
prefixes, features, capacity and future budgets, retaining tail-label uncertainty.
Termination already uses episode credit: swapping to G alone is not a new answer.
Do not fit on these revealed evaluation paths; reserve a new development holdout
before repair. No further jobs were submitted after reading these results.

These are development diagnostics with2 roots/task and4 probe paths/root, not
confirmatory CIs. Information benchmarks change one sampled decision, cannot be
summed into a deployed-policy gain, and cover only sampled bin-start opportunities.
Conditional latent-state results are not observable-state Bayes bounds. Unknown
earlier interrupted costs and the parent competence failure still block C1/C2.

Sources: [summary](../results/plan_validity_local_episode_diagnostic_20261110/summary.json),
[manifest](../protocols/plan_validity_local_episode_diagnostic_2026-11-10.json),
[completion](../results/plan_validity_local_episode_diagnostic_20261110/inputs/completion_receipt.json).
