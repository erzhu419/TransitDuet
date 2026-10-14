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

- t152346-t152377 analyses and t152378 report all exited 0; six tests passed. All 119,040 frozen-choice and 476,160 physical-score comparisons reproduced exactly. Pulled 1,699,680 aggregate bytes; no models/panels, new fitting or rollouts.
- Predeclared original-anchor rank8 focus, held-prefix/opposite-future outcomes (increments versus the original action; smaller is better):

| Intervention | Cost | Wait (min) |
| --- | ---: | ---: |
| Base / zero residual | -0.00547212 | +0.00872083 |
| Cost-only | -0.00659011 | -0.00232500 |
| Wait-only | -0.00537694 | +0.00933750 |
| Joint | -0.00668290 | -0.00350833 |

- In this cell, cost-only supplies 92.3% of the joint cost gain and 90.3% of the joint wait gain versus base. Wait-only worsens both; interaction contributes -0.00018796 cost / -0.00180000 min. Adding the wait head after cost contributes -0.00009279 / -0.00118333 min and switches 7/60 actions. The gain is mainly cost ranking with a smaller beneficial interaction, not a standalone wait-head gain.
- Root401 wait is already worse than raw in base (+0.00418333 versus -0.00485000 min). Cost-only recovers most of that deficit (-0.00399167); joint only slightly improves it (-0.00404167). The remaining deficit is not created by the learned wait residual. Pooled focus joint also has higher realized unsafe fraction than raw (21.7% versus 15.0%), despite better mean cost/wait.
- This is not a uniform improvement: fixed rank8 joint beats raw on both axes in only 1/4 mixed_a future directions and 0/4 mixed_b directions. In the same focus cell's training-prefix/opposite-future scope, cost-only improves cost but worsens wait by +0.00430000 min versus base. Default remains unchanged.

## Next

Freeze the cost residual and cross it with the existing raw wait-logit gate versus the physical wait gate. Require rank0/raw-gate to reproduce raw exactly, keep all ranks/anchors and the inherited joint selection, and score the same source/future splits without refitting. This separates a useful learned ranking from damage introduced by replacing the existing feasible menu before increasing model complexity.

## Limitations

Predictor-head interventions on reused one-action suffix outcomes are development attribution, not independent-seed CI, environment-level causal mechanisms or end-to-end HRL proof. Feature singular modes are not frequency bands.
