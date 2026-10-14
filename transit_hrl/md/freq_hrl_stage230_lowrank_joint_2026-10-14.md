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

Pending. Determine whether retaining dominant modes improves held-prefix wait prediction, whether cost refitting harms the matched control, and whether either selected arm beats raw on both physical metrics before changing defaults.

## Limitations

Development cohort and future partitions are reused. Feature-variance modes are not frequency bands. This is a supervised one-action suffix diagnostic, not independent-seed CI or end-to-end HRL evidence.
