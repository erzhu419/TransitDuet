# Stage232: frozen cost ranking crossed with wait gates

## Protocol

- Run: `native_transit_cost_gate_stage232_development_20261014_r1`.
- Reuse all 64 Stage230 cost paths and 32 raw references; retain the 240-state/eight-draw development cohort, original/raw training anchors, prefix folds and opposite-future halves. No fitting, SVD, new episodes, actor updates or weight artifacts.
- Four arms at the same rank 0, 1, 3, 8, 16: physical cost/physical wait, learned cost/physical wait, physical cost/raw logits, learned cost/raw logits. Neither gate uses the learned physical wait residual.
- Retain each anchor's existing identity-first action priority and inherited joint MSE/selected rank. Raw logits remain original-action feasibility scores, not rebased additive wait increments. Raw choice is already eligible; rank0/raw gate must reproduce raw in both anchors. When the joint selector holds raw, all four arms hold raw.
- Reproduce Stage231 base/cost controls and physical scores in all four train/held, same/other-future scopes. Report ranking effects under each gate, gate effects at each ranking, interaction, unsafe outcomes and feasible-menu/action changes; keep all32 cells and the predeclared `mixed_b/fold1/even_odd/direction0` rank8 focus.
- 32 scheduler analyses plus one report, unbound node001-node006, one CPU/thread each. Pull only aggregate JSON; models/panels remain remote.

## Results

- t152380-t152411 analyses and t152412 report all exited 0; six tests passed. All 345,600 choice and 691,200 physical-score comparisons reproduced exactly, including rank0/raw-gate identity under both anchors. Aggregate pull: 1,720,783 bytes; no panels/weights or new fits/rollouts.
- Predeclared original-anchor rank8 focus, held-prefix/opposite-future outcomes (increments versus the original action; smaller is better):

| Controller | Cost | Wait (min) | Wait-worsening actions |
| --- | ---: | ---: | ---: |
| Raw | -0.00370825 | -0.00300417 | 15.0% |
| Learned cost / physical gate | -0.00659011 | -0.00232500 | 23.3% |
| Learned cost / raw gate | -0.00401957 | -0.00551667 | 13.3% |

- The new arm changes 7/60 held actions versus raw and improves both means at each focus source root. Root401 cost/wait becomes -0.00411682/-0.00820000 versus raw -0.00369682/-0.00485000; wait-worsening fraction stays 16.7%. Root397 cost/wait also improves, and that fraction falls from 13.3% to 10.0%. Thus the prior source wait/unsafe deficit can be removed by the existing menu while retaining a smaller genuine learned-cost gain.
- This does not make fixed rank8 generally superior: in mixed_a/mixed_b, new-arm held-other cost is worse than raw by +0.00079368/+0.00090506, although wait and wait-worsening fractions improve. In the focus held-same-future scope, cost is worse by +0.00140435; on training-prefix/opposite-future, wait is worse by +0.00358750 min. Gate substitution helps, but cost ranking still has prefix/future-dependent tradeoffs.
- Inherited original-anchor selection gives mixed pooled cost/wait -0.00225963/-0.00461589 versus raw -0.00224017/-0.00445885, with wait-worsening fraction 14.90% versus 15.00%. Only one of 16 mixed cells uses a nonzero rank; the other cells hold raw or select zero. Single-source root397 with inherited selection worsens both means. Default remains unchanged.

## Next

Do prefix-and-future inner selection specifically for learned cost under the raw gate, retaining the frozen outer references/paths and rank0/raw-hold controls. Select using the actual new controller's physical cost, wait and wait-worsening outcomes at each source root, not the old joint physical-gate score or outer held-out gains. Reuse the current development cohort; do not increase model capacity or open the reserved confirmation cohorts.

## Limitations

These are frozen predictor/menu interventions on reused one-action suffixes, not independent-seed CI or end-to-end HRL evidence. The inherited rank selector was fitted for the prior physical joint controller, not optimized for the new raw-gate arm.
