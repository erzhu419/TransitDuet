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

## Run Receipt

Implementation `5549f583d5`, registration `01154fb6ec`; eight tests passed.
All12 learned source uppers were read on node004 with matching source fits.
Run `pointmaze_continuation_replication_stage131_frozen_20261008_r1`:
`t136526-t136531` accepted, ordered by the six roots above. Task specifications,
fresh seed roles, new/inherited budgets and scheduler receipt are saved with the run.

## Completed Result

All six tasks completed;37,824new episodes/45,388,800steps,187,402bytes fetched.
Six-endpoint corrected root-cluster bootstrap intervals:

| Period | Refresh minus forecast | Refresh minus single | Refresh minus stale |
|---|---|---|---|
|50|+1.438688 [1.052682,1.667235]|+0.485376 [0.185617,0.663596]|+0.071236 [-0.100068,0.266523]|
|100|+0.688348 [0.421125,0.878955]|+0.330307 [0.224276,0.418890]|+0.110088 [0.034931,0.170537]|

Continuation increments replicate at both periods; relabeling replicates at100,
not universally. Material forecast gain closes at50 only. Keep root410049's
period50 increment-0.012623, its negative A-to-B crossfit-0.049395, and
root410037's period100 refresh-minus-stale-0.010491. Five of six period100
calibration fits hit scale1 with positive fitted endpoint slope. Return to the
two development roots to test a third local update; do not merge development
roots into this CI or lower0.5. This test cannot erase the Stage131 result.

## Limitations

These source policies were used in earlier research and the one-step replication;
only continuation development excludes them. This is internal frozen-method
replication, not an untouched external benchmark. Per-update geometry matches,
not cumulative distance from zero. Joint-HRL, promotion and frequency-specific
claims remain separate.
