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

- t152248-t152279 fits and t152280 report all exited 0; 28 tests passed. All 1820 frozen-control comparisons and inner raw references reproduced exactly. Pulled report: 1,452,677 bytes; raw panels/model weights: zero.
- Final designs have numerical rank 149-150 but participation effective rank about 3-3.5; 8-9 modes contain 95% of feature energy. Strong gamma=1000 has about 0.187 effective df/head, versus 34-36 at gamma=0.01. Nominal parameter count substantially overstates independent feature directions.
- Equal-weight four-view raw-anchor held-other MSE: zero residual **1.239783**, weak ridge **1.924226**, strongest ridge **1.239732**, inner-MSE-selected ridge **1.243683**. Explicit shrinkage controls overfit but gives no meaningful aggregate prediction gain. MSE selection chooses zero in 17/32 original and 14/32 raw cells; most other selections strongly shrink the residual.
- Under the original anchor, weak ridge wait error improves on a new future within training prefixes in all four views (pooled **-7.86%**) but worsens on held prefixes in all four (**+57.00%**). Cost error worsens within training prefixes on new futures in all four (pooled **+14.13%**). There is within-prefix wait signal; neither complete signal absence nor missing nonlinear capacity alone explains the failure.
- Joint-selected ridge original/raw are negative on cost and wait versus the original action in 8/8 mixed future directions. Raw-anchor ridge beats raw on both axes in **0/8**; 15/16 mixed fitting cells hold raw. Its mixed pooled cost/wait **-0.00223780/-0.00401120 min** is slightly worse than raw **-0.00224017/-0.00445885 min**. Root401 original-anchor gains in 2/4 directions remain source-local.

## Next

Keep the default unchanged. Compare a frozen-cost, training-only low-rank wait residual against a matched dual-head low-rank residual. This isolates unnecessary cost refitting from fitting weak feature modes and tests held-prefix transfer, rather than expanding the ridge grid or training longer.

## Limitations

Same 240-state/eight-draw development cohort and overlapping future partitions. These are one-action suffix diagnostics, not independent-seed CI or end-to-end learned HRL evidence.
