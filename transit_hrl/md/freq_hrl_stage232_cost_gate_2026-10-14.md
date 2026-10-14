# Stage232: frozen cost ranking crossed with wait gates

## Protocol

- Run: `native_transit_cost_gate_stage232_development_20261014_r1`.
- Reuse all 64 Stage230 cost paths and 32 raw references; retain the 240-state/eight-draw development cohort, original/raw training anchors, prefix folds and opposite-future halves. No fitting, SVD, new episodes, actor updates or weight artifacts.
- Four arms at the same rank 0, 1, 3, 8, 16: physical cost/physical wait, learned cost/physical wait, physical cost/raw logits, learned cost/raw logits. Neither gate uses the learned physical wait residual.
- Retain each anchor's existing identity-first action priority and inherited joint MSE/selected rank. Raw logits remain original-action feasibility scores, not rebased additive wait increments. Raw choice is already eligible; rank0/raw gate must reproduce raw in both anchors. When the joint selector holds raw, all four arms hold raw.
- Reproduce Stage231 base/cost controls and physical scores in all four train/held, same/other-future scopes. Report ranking effects under each gate, gate effects at each ranking, interaction, unsafe outcomes and feasible-menu/action changes; keep all32 cells and the predeclared `mixed_b/fold1/even_odd/direction0` rank8 focus.
- 32 scheduler analyses plus one report, unbound node001-node006, one CPU/thread each. Pull only aggregate JSON; models/panels remain remote.

## Results

Pending. Determine whether the existing raw menu retains learned cost gains and removes the residual wait/unsafe deficit before adding capacity or retraining.

## Limitations

These are frozen predictor/menu interventions on reused one-action suffixes, not independent-seed CI or end-to-end HRL evidence. The inherited rank selector was fitted for the prior physical joint controller, not optimized for the new raw-gate arm.
