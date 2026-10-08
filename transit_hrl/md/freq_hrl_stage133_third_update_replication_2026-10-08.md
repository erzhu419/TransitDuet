# Stage133: Frozen Third Update Replication

Stage132 passes its two-root development gate: third-minus-second gains
+0.641560/+0.373860 and forecast gains+1.998440/+1.159728 at50/100.
Freeze the same kernel; no fourth step or period-specific method selection.

Run roots410037/410049/410061/410073/410089/410101 from their completed Stage131
refresh upper weights (two learned steps), not from development-root weights.
Four new current-policy label scenes/two panels,16 new calibration scenes/two
panels and64 new evaluation scenes. Same26-dimensional projection,epsilon0.005,
damping1,per-update radius0.000555556,std0.15,reference authority0.05 and frozen
lower/critics. Both periods and all seven Stage132 controls retained.

Six primary endpoints: refresh-minus-forecast/two_step/radial-control at50/100.
Root-cluster bootstrap65,536draws,Bonferroni6,familywise alpha0.05 within this
stage. Material gate requires forecast CI lower>0.5 and incremental CI lower>0
at both periods. Relabeling value is reported separately per period; universal
relabeling needs both corrected CI lower>0. Keep all roots and negative results.

New cost/root:4,896label+320calibration+192crossfit+896evaluation=6,304episodes/
7,564,800steps. Six roots:37,824episodes/45,388,800steps. Immediate Stage131 cost
separate; previous training and source-chain costs remain inherited, not zero.
Six17CPU/12GiB tasks,16workers each,dynamic node001-node006,no hard node pin or
waiting aggregation task. Four final upper weights/root stay server-side;
pull compact JSON only.

## Run Receipt

Implementation `506b981ebc`, registration `a661716dd0`;13 unique test cases
passed, including the corrected budget's native-path retest. All12 two-step
source upper checkpoints were verified on node004 without downloading weights.
Run `pointmaze_third_update_replication_stage133_frozen_20261008_r1`:
`t136544-t136549` accepted, ordered by the six roots above; initial snapshot
queued. Exact task specifications, seed roles and budgets are preregistered;
compact dispatch receipt saved with the run. No result or CI is available yet.

## Limitations

These six roots were used in earlier experiments and two previous replications;
this is internal frozen-method replication, not untouched external confirmation.
Intervals correct six endpoints within this stage, not the adaptive research
sequence. Radial continuation shares the two-step source and is not a fully
stale three-step training route. Per-update radius matches, not cumulative
distance from zero. Stage131 findings and joint-HRL/promotion/frequency-specific
claims remain unchanged.
