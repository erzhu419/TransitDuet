# Stage131: Frozen Compact Continuation Replication

Stage130 exceeds0.5 forecast gain and improves the single-step policy at both
periods. Refresh-minus-stale is+0.249798 at100 but-0.004481 at50; retain this
boundary instead of selecting a different method per period.

Freeze the Stage130 algorithm and run the same kernel on six source roots
410037/410049/410061/410073/410089/410101, starting from their Stage129 compact
weights. Four fresh current-policy label scenes/two noise trajectories,16 fresh
calibration scenes/two panels and64 new evaluation scenes. Same projection,
epsilon0.005, damping1, per-update Fisher radius0.000555556, variance and0.05
authority; frozen lower/critics. Keep both periods and all seven controls.

Six primary endpoints: refresh-minus-forecast/single/stale at50/100. Root-cluster
bootstrap65,536draws, Bonferroni6, familywise alpha0.05. Material continuation
requires forecast CI lower>0.5 and single CI lower>0 at both periods. Relabeling
CI lower>0 is reported per period; universal relabeling needs both to pass.
No root filtering, period-specific method switch or evaluation-based tuning.

New cost/root:4,896label+320calibration+192crossfit+896evaluation=6,304episodes/
7,564,800steps. Report inherited Stage129 cost separately. Six tasks with
16workers+parent/12GiB, dynamic node001-node006; no waiting aggregation job.
Only compact JSON pulled; four final upper weights/root stay server-side.

## Limitations

These source policies were used in earlier research and the one-step replication;
only continuation development excludes them. This is internal frozen-method
replication, not an untouched external benchmark. Per-update geometry matches,
not cumulative distance from zero. Joint-HRL, promotion and frequency-specific
claims remain separate.
