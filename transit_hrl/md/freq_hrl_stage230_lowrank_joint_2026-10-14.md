# Stage230: matched low-rank wait and joint residuals

## Protocol

- Run: `native_transit_lowrank_joint_stage230_development_20261014_r1`.
- Same 198 causal features, physical targets, training-only RMS, original/raw anchors, 240-state/eight-draw cohort and prefix/future splits. Freeze all 29 Stage229 controllers.
- Training-only truncated SVD at rank 0, 1, 3, 8, 16. Each original/raw path shares one SVD and the exact same wait coefficients between two arms: frozen-cost wait residual, and dual-head cost/wait residual. Retain only numerically nonzero singular directions.
- Rank is selected using only held-prefix/opposite-future validation inside the outer training half. Pool physical means before root normalization; ties prefer zero then lower rank; keep raw if no level strictly improves both axes at each source root.
- Record per-head errors at every validation rank and fixed/MSE/joint control metrics across all four scopes. Verify identical wait predictions at fixed rank and unchanged cost predictions in the wait-only arm. Frozen cost prediction does not guarantee unchanged realized cost.
- 32 fits plus one report, unbound node001-node006, one CPU/thread each. 640 clean raw refits (384,000 updates), 1344 shared residual SVD paths and 5376 positive-rank solutions; no residual optimizer updates, new native episodes or RL actor updates.
- Models/panels stay remote; pull only the aggregate report. The matched arms share preprocessing and wait fits; grid/compute is not equal to frozen earlier methods.

## Results

- t152281-t152312 fits and t152313 report all exited 0; 36 tests passed. All 2772 frozen-control comparisons, 320 matched-arm checks and inner raw references reproduced exactly. Pulled aggregate: 2,044,043 bytes, plus a small two-cell audit; no raw panels or weights.
- Raw-anchor four-view held-other wait MSE: zero **1.185213**, rank3 **1.182237**, rank8 **1.234040**, rank16 **1.359608**. Low rank controls weak-mode overfit; the rank3 gain is small. Under the original anchor, higher ranks improve within-training-prefix new-future wait error but still worsen held-prefix error.
- Original-anchor joint-selected mixed cost/wait **-0.00243980/-0.00465755 min**, versus raw **-0.00224017/-0.00445885 min**: both pooled increments improve, 8/8 directions remain negative versus the original action, and 2/8 strictly beat raw on both axes. Of 16 mixed cells, 14 hold raw, one selects zero residual and one rank8. Raw-anchor joint-selected still has no strict dual-metric win over raw in mixed directions.
- The nonzero cell `mixed_b/fold1/even_odd/direction0` genuinely changes 16/60 held actions across eight unseen prefixes. Rank8 held-other cost/wait **-0.00668290/-0.00350833 min** versus the same cell's zero residual **-0.00547212/+0.00872083 min**; both improve. It also beats pooled raw **-0.00370825/-0.00300417 min**, but root401 has a wait tradeoff versus raw. The positive result is not entirely zero-residual routing, nor is it a uniform-source gain.
- Frozen-cost wait-only does not dominate joint control. At the same rank both arms have identical wait predictions, yet realized controls differ; prediction MSE alone does not establish which cost ranking is useful. Default remains unchanged.

## Next

Freeze these paths and add the missing cost-only counterfactual, completing baseline/cost-only/wait-only/joint at identical ranks. Attribute the rank8 improvement to cost ranking, wait feasibility or their interaction across future partitions and source roots without further fitting or new cohorts.

## Limitations

Development cohort and future partitions are reused. Feature-variance modes are not frequency bands. This is a supervised one-action suffix diagnostic, not independent-seed CI or end-to-end HRL evidence.
