# Stage-23 Contextual Paired-Value Result

**The two-root development gate failed.** Preflight `t101514` completed on
node004; full tasks `t101516/101517` completed on node004/node005 at `eef7b5394c`.
Full results total 45,179 bytes. Each root completed 16 closed-form fits on
the original labels, with zero new environment steps or controller training.
Twenty-two focused tests passed; scoring and fold/sample accounting agree.

| Root | Corrected zero MSE | Frozen linear | Contextual | Random context |
|---|---:|---:|---:|---:|
| 209011 | 0.002280754 | 0.001908796 | 0.001897865 | 0.001905474 |
| 209061 | 0.001247787 | 0.001396302 | 0.001387436 | 0.001372960 |

Contextual improves only 0.57%/0.63% versus linear. Root 209011 passes the
point gate; 209061 remains 11.19% worse than zero and 1.05% worse than random
context. Thus this bounded interaction has not established useful contextual
specificity across both roots. Mean effective degrees of freedom are
2.056/2.023 for contextual and 2.049/2.074 for the matched control.

Next: diagnose the decision value of tail credit relative to short-window
credit before another critic expansion. Use disjoint cached future subsets
for oracle-tail action selection and scoring, avoiding same-label oracle
selection optimism. This is an oracle-reference diagnostic, not a
replacement gate for the failed candidate. Retain Stage-9; no deployment,
alpha tuning or seed expansion follows from Stage-23.

## Limitations

Scores reuse development labels and are not independent confirmation or
policy-performance results. Parameter count and row norms match the random
control, but effective capacity is not identical. Failure of this interaction
basis does not establish that all causal context is uninformative.
