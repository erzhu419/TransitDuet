# Stage231: frozen cost/wait factorial attribution

## Protocol

- Run: `native_transit_factorial_stage231_development_20261014_r1`.
- Reuse all 64 Stage230 paths and 32 raw references; same state/future cohort and original/raw anchors. No fitting, SVD, actor updates, new episodes or new weight artifacts.
- Complete base/cost-only/wait-only/joint at the same rank 0, 1, 3, 8, 16. Cost-only changes ranking while holding the physical wait feasibility predictor fixed; wait-only changes that predictor while holding ranking fixed.
- All four interventions use the existing joint MSE/selected rank. No cost-only selection or held-out rank search; when the joint selector holds raw, all four hold raw.
- Reproduce stored raw and fixed-rank base/wait/joint choices and physical outcomes at each future half. Report additive contrasts and interaction `joint - cost-only - wait-only + base`, feasible-menu changes and action switches by source/future.
- Predeclared focus: `mixed_b/fold1/even_odd/direction0`; retain all32 cells rather than only the positive case. All four train/held and same/other-future scopes remain reported.
- 32 analysis tasks plus one report, unbound node001-node006, one CPU/thread each. Pull only the aggregate report; models/panels remain remote.

## Results

Pending. Determine whether cost ranking, wait feasibility or their interaction explains the genuine rank8 improvement and the source-level tradeoff, then target the next optimization accordingly.

## Limitations

Predictor-head interventions on reused one-action suffix outcomes are development attribution, not independent-seed CI, environment-level causal mechanisms or end-to-end HRL proof. Feature singular modes are not frequency bands.
