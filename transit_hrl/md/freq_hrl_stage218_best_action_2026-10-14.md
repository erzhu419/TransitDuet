# Stage218: matched roots and best-action constraints

Stage217 mixed-root full-order failures did not isolate sample size or the weaker objective of choosing an optimal action. Stage218 uses only the original 240 mean8 development states.

- Four views: root397, root401, mixed A and complementary mixed B. Each has 120 states, 20 complete prefixes and four prefixes per regime. Mixed A uses root397 replicas0/1 and root401 replicas2/3; B reverses this assignment.
- Both existing future partitions are retained. On the common true-safe menu, use only states whose identity-first best action agrees between halves. Report disagreements and preceding near-ties separately.
- Compare all stable orders, best-action inequalities, and their union on exactly the same eligible states. Keep wait-sign feasibility as a separate control. Normalization uses each selected view's causal inputs only.
- Scheduler: one CPU, 3072 MB, unbound node001-node006; at most32 linear programs, no new native episodes or actor updates. Pull only the compact report.

If best-only is feasible after matching but full orders are not, the next step is linear decision-focused learning, not an assertion that nonlinear conditioning is necessary. If mixed-root best-only remains infeasible while single-root is feasible in both partitions and complementary views, test causal conditional features next.

## Result

`t151836`, node004, exit0: all32 bounds numerically feasible, unresolved0. This includes full orders and wait signs in both single-root and both mixed-root views. Matching input count removed the previous qualitative single/mixed difference; weakening the target was not needed to make these120-state views feasible.

Best targets cover82-92/120 states, with23-34 nonidentity optima;28-38 states disagree between future halves, near-tie exclusions0. No actor updates or new episodes. Only38,676 bytes of report were pulled;14 tests passed. Next: test best-only on the complete240-state cohort before selecting a larger model.

## Limitations

These are finite training-class diagnostics under an oracle-safe menu, not held-out learned-policy evidence. Feasibility uses numerical tolerance1e-7, without coefficient-norm minimization; LP witnesses are not deployment parameters. Eligible counts differ despite equal input counts; both future partitions reuse the same eight draws. Root bundles include different lower policies and demand/prefix seeds.
