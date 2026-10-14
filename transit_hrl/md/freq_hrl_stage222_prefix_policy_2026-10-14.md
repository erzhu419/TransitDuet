# Stage222: prefix/future and selected-action calibration

Stage221's prediction-only shrinkage reduced noise overfit but lost mixed_b wait gains. This stage reuses all final Stage221 models and compares prefix/future risk-only shrinkage against the same objective constrained by selected-action cost/wait floors, separately for each source root.

Ten leave-one-prefix folds crossed with two future splits per cell: 54 training states/two draws; six validation states/other two draws. Budget and protocol: `results/native_transit_prefix_policy_stage222_development_20261014_r1/preregistration.json`.

## Results

Fits t151936-t151967 and report t151970 completed with exit0. Eight new tests and six Stage221 tests passed. All 32 cells qualified; 1,584 control comparisons reproduced exactly. Only the 516,903-byte report was pulled, no raw/models. Default/native policy unchanged.

Prefix/future risk-only conditioned alpha averages 0.047275 versus Stage221 future-only 0.151464. Held physical wait mass-risk falls to 0.04196498 (raw 0.06078162), but remains 0.98% above GLM. Adding prefix holdout further attenuates the residual; risk-only shrinkage still loses mixed_b wait benefit.

Constrained conditioned calibration lowers held mass-risk 16.0% versus raw, and restores joint negative cost/wait in 7/8 mixed/future summaries versus 4/8 prediction-only shrinkage. Raw remains 8/8. The remaining mixed_b even_odd/0 wait is +0.00328125 versus raw -0.00152708. No universal no-tradeoff claim.

Final train_same wait worsens versus raw in all four views despite satisfied inner OOF floors. This failure appears before outer prefix/future transfer. The inner two-draw/54-state models and final four-draw/60-state models are not the same fitted controller.

## Next Decision

Keep the raw conditioned control. Next match inner/final target precision and fitting protocol to test constraint transfer, rather than tune alpha or blame outer OOD alone.

## Limitations

Inner training uses nine prefixes/two draws versus ten/four in the final model. Floors are relative OOF means, not held-control guarantees. The same 240 states/eight futures are reused; roots bundle lower-policy and demand changes. This is still one-action supervised development, not independent-seed CI or end-to-end HRL.
