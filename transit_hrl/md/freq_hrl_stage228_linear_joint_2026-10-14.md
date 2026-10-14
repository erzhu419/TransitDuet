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

## Decision

Pending execution. Compare training/other-future/outer-held errors, selected checkpoint distributions, and cost/wait versus raw and frozen nonlinear paths. Adopt no new default from in-sample or single-source gains alone.
