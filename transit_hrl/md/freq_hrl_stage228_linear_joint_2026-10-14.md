# Stage228: low-capacity joint residual

## Protocol

- Run: `native_transit_linear_joint_stage228_development_20261014_r1`.
- Test whether reducing capacity improves physical cost/wait generalization after Stage227 repaired checkpoint aggregation.
- Two linear heads over the same 198 engineered causal action features: 396 parameters versus frozen nonlinear 3410. Linear bias and 12 action-constant context terms cancel exactly under either action anchor; nonlinear context interactions remain in the comparator.
- Same original/raw anchors, physical targets, own-training RMS scales, AdamW learning rate/decay, 600 steps and six snapshots. This is not a plain-feature linearity test.
- Same 240-state/eight-draw development cohort, matched 120-state views, outer prefix split and two future-half partitions. No new confirmation cohort.
- Inner references are refit on 54 states/two draws; validation excludes the entire prefix and uses the opposite inner future half. Final linear paths use 60 states/four draws.
- Pool physical validation cost/wait means before positive source-root normalization. Select an earlier tied checkpoint; hold raw if no checkpoint improves both axes at every source root.
- Freeze all 11 Stage227 controllers; add linear zero/last/MSE/joint-selected controls under both anchors. Reproduce old decisions and prediction errors.
- 32 fitting tasks plus one report, unbound node001-node006, one CPU/thread per task. 640 clean raw refits, 1280 inner plus 64 final linear paths; 1,190,400 optimizer updates, zero native episodes or RL actor updates.
- Model/panel artifacts stay remote. Pull only the compact aggregate report.

## Results

- t152214-t152245 fits and t152246 report all completed with exit 0. Twenty tests passed. All 1056 frozen-control comparisons and clean inner raw references reproduced exactly. Pulled aggregate report: 662,106 bytes; no raw panels or model weights.
- Equal-weight four-view raw-anchor normalized MSE: zero residual train/held-other **0.820/1.240**; nonlinear last **0.036/2.865**; linear last **0.518/2.223**. Lower capacity reduces held error 22.4% versus nonlinear last, but remains 79.3% above zero residual. Final fit ledgers match recorded training errors.
- Internal MSE selection keeps step 0 in 30/32 original-anchor and 31/32 raw-anchor cells. Linear last is worse than zero residual on held-other in all four views.
- Joint-selected linear original/raw restore negative cost and wait versus the original action in 8/8 mixed-view future directions; frozen nonlinear gives 7/8 and 6/8. Linear raw is strictly better than raw on both axes in **0/8** directions; 15/16 mixed fitting cells hold raw.
- Mixed pooled raw cost/wait: **-0.00224017/-0.00445885 min**; linear raw joint-selected: **-0.00216232/-0.00429453 min**, both slightly worse. The sole non-hold mixed raw cell passes physical inner selection but harms held cost/wait. Selection arithmetic is correct; its improvement does not generalize.

## Next

Keep the default controller unchanged. Test explicit ridge shrinkage and the effective rank of the same paired feature design; distinguish weak regularization from insufficient learnable residual signal without increasing network size or training duration.

## Limitations

The two future partitions reuse the same development draws. Negative increments versus the original action are not gains versus raw; source-local gains do not establish cross-source improvement. These supervised one-action suffix diagnostics are not a new-seed CI or end-to-end learned HRL result.
