# Stage229: explicit ridge shrinkage and effective rank

## Protocol

- Run: `native_transit_ridge_joint_stage229_development_20261014_r1`.
- Same 198 engineered causal features, paired physical cost/wait targets, own-training RMS, original/raw anchors and clean prefix/future validation as Stage228. Freeze all 19 previous controllers.
- Solve residual coefficients by training-only SVD. Minimize mean paired/head squared error plus lambda times coefficient Frobenius norm squared divided by two heads. Lambda is gamma times training Gram trace/198; gamma grid: 1000, 100, 10, 1, 0.1, 0.01, plus exact zero residual.
- Pool physical validation means before source-root normalization. Choose MSE or worst-root cost/wait score; ties prefer zero then stronger shrinkage. Joint selection holds raw if no level strictly improves both axes at every source root.
- Record training-only numerical rank, participation effective rank, 95% energy rank and ridge effective degrees of freedom per head. Separate normalized cost and wait errors in all evaluation scopes.
- 32 fits plus one report on unbound node001-node006, one CPU/thread each. 640 clean raw refits (384,000 updates), 1344 residual SVD paths and 8064 positive ridge solutions; zero residual optimizer/RL updates or new native episodes.
- This changes the optimizer and regularization grid, not only capacity; no equal-compute claim against frozen gradient paths. All raw/model artifacts stay remote; pull only the aggregate report.

## Results

Pending. Compare ridge versus zero residual, frozen linear and nonlinear errors, selected shrinkage/df, and held physical cost/wait versus raw before changing any default.

## Limitations

Same 240-state/eight-draw development cohort and overlapping future partitions. These are one-action suffix diagnostics, not independent-seed CI or end-to-end learned HRL evidence.
