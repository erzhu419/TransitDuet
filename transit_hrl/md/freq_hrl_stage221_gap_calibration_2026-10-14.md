# Stage221: cost-gap objective and train-future wait shrinkage

Same Stage220 cohort, folds and controls. The new 98-parameter ridge-residual learns soft physical cost gaps rather than hard safe winners. Three wait residuals get identical 2/2-future inner fits and convex alpha shrinkage; outer held labels never choose parameters. Full protocol and budget are in `results/native_transit_gap_calibration_stage221_development_20261014_r1/preregistration.json`.

## Results

Tasks t151918-t151925 and t151928 completed with exit0; 12 tests passed. All 32 cells and budgets qualified; 640 control-summary comparisons reproduced exactly. Only the 469,069-byte report was pulled, no raw/models. The default/native policy is unchanged.

With the same linear gate, gap-minus-CE held cost is -0.00143002, -0.00053605, -0.00399175, -0.00199130 for root397/root401/mixed_a/mixed_b. All improve, but only two beat ridge and only root401 also improves wait over CE.

Conditioned physical wait mass-risk: train_other 0.04771256 -> 0.03356002; held_other 0.06078162 -> 0.04272991 (-29.7%). Held risk still exceeds GLM (mean 0.04155699) in every view: future-noise overfit improves, prefix generalization does not close.

Ridge + conditioned mixed_b wait: -0.00269740 -> +0.00268698. Joint negative cost/wait across eight mixed/future summaries drops 8/8 -> 4/8 (gap raw 6/8; shrunk 5/8). Better calibration is not better control.

## Next Decision

Keep ridge + raw conditioned as control. Next: training-internal prefix AND future holdouts, plus an explicit selected-action wait/cost constraint rather than prediction log-loss alone.

## Limitations

Inner fitting uses two draws versus four in the final fit. Sensitivities reuse eight draws; roots bundle lower-policy and demand/prefix changes. This remains supervised one-action validation with a frozen greedy suffix, not independent-seed CI or end-to-end HRL.
