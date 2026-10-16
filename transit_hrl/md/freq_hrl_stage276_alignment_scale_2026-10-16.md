# Stage276: Residual Scale versus Normalized Alignment

t166019/node002: done/0; 14 tests passed. Zero fits, inference or simulations;
536 kB scalar pull. Stage275 replay exact; partition/product errors <2.11e-19.

Primary mixed pooled, paired, opposite16. Alignment = 2 * mass * correction-RMS
* residual-RMS * uncentered cosine. Symmetric 24-order contributions:

| Factor | Alignment-gap contribution, 1e-6 | Share of net gap |
| --- | ---: | ---: |
| Normalized alignment | +46.344 | 89.05% |
| Raw residual RMS | +6.264 | 12.04% |
| Correction RMS | -0.567 | -1.09% |
| Population mass | 0 | 0% |

Dominant prefixes intersecting probe8/9: 28 repeated comparisons contribute
+41.481e-6 (79.71% of net gap); excluded/held cosine is +0.192 / -0.619.
The same prefixes at other probes have 32 comparisons, cosine -0.330 / -0.324,
and contribution -0.856e-6. Uniformly shrinking corrections is not the priority.

Next Stage277: a matched-capacity causal-state-conditioned residual fit, same masks,
banks/clip and train-only normalization. New residual fits needed, not started here;
no raw-reference retraining, simulation or ID gate.

Limitations: development-selected groups, finite16 targets and historical reference
conditioning; not causal/seed-CI/policy evidence. Bank1 assigned-bank scale/alignment
effects are comparable; root397 dominant subset reverses. Scheduler timing invalid.
