# Stage218/219: matched samples and optimal-action constraints

## Protocol

Stage218 compares root397, root401 and complementary mixed A/B, each with120 states,20 complete prefixes and four prefixes per regime. A uses root397 replicas0/1 and root401 replicas2/3; B reverses this. Stage219 restores all240 states as a sample-growth reference.

Both future partitions use the same causal features. Best targets require half-group agreement on the common true-safe menu, with identity-first ties. Compare best-only against full orders on exactly the same target states, plus full-data order and wait-sign controls.

## Results

- `t151836`: all32 matched-size bounds feasible, including full orders and wait signs. Input-count matching removes the earlier qualitative single/mixed difference. Best targets cover82-92/120 states; near-tie exclusions0.
- `t151841`: full240 cost orders infeasible, but best-only AND same-target full orders feasible in both partitions. Stable targets cover174/180 states with50/57 nonidentity optima;66/60 states disagree between halves. Excluding disputed targets, not weaker ordering alone, suffices for feasibility.
- Full240 wait-sign bounds remain infeasible: minimum mean margin deficits0.077144/0.090511, dimensionless. The linear cost head can express stable optima; exact full-data linear safety classification remains limited.

Both tasks: node004, exit0;40 bounds qualified, unresolved0; no new episodes or actor updates.17 tests passed. Reports pulled:38,676+13,414 bytes, no raw data/models.

## Next

Train regularized decision-focused cost heads using one future half only; evaluate the other half on all held prefixes, including disputed states. Separately compare linear and causal conditional wait-risk heads at matched budgets.

## Limitations

Numerical training-class feasibility (tolerance1e-7), not learned-policy generalization. Oracle safe menus, no coefficient-norm minimization, reused eight draws. Root bundles change lower policy and demand seeds together.
