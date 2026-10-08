# Stage132: Third Local Policy Update

Stage131 confirms a second update at both periods and relabeling value at100.
The period100 forecast gain+0.688348 has corrected CI [0.421125,0.878955],
so the0.5 material gate remains open. Five of six long-period calibration fits
still hit scale1 with positive fitted endpoint slope; test another update rather
than merge development roots or relax the gate.

Return to roots410011/410023, starting from Stage130 refresh weights (two steps).
Use the same update kernel and four new current-policy label scenes/two panels,
16 new calibration scenes/two panels,32 new evaluation scenes. Same26-dimensional
projection, epsilon0.005, damping1, per-update radius0.000555556, variance and0.05
authority; frozen lower/critics. Both periods and all seven controls remain.
Compare refreshed credit against radial continuation of the complete two-step
parameter displacement from zero, at the same current-state radius and budget.
The baseline is explicitly named two_step, not single.

New cost/root:5,856episodes/7,027,200steps, inherited Stage130 cost separate.
Two17CPU/12GiB tasks on dynamic node001-node006. Four final weights/root stay
server-side; no raw states/traces pulled and no waiting aggregation job.

Require positive fresh-evaluation increments over two_step and material forecast
gains at both periods before another frozen replication. Report radial-control
differences separately; negative results stop a claim of benefit from this step.

## Run Receipt

Implementation `e811b9eaac`, registration `1bd180103b`; ten tests passed.
All four two-step source uppers were read on node004 with matching source fits.
Run `pointmaze_third_update_stage132_pilot_20261008_r1`:
`t136539` root410011 and `t136540` root410023 accepted. Task specifications,
fresh seed roles, new/inherited budgets and scheduler receipt are saved with the run.

## Completed Result

Both tasks completed on node004/node006. New cost:11,712episodes/14,054,400steps;
46,967bytes fetched, excluding labels and checkpoints. Budget, source, fresh
paired rosters, frozen lower/critics and blinded forecast identity match.

| Period | Refresh minus forecast | Refresh minus two_step | Refresh minus radial control |
|---|---|---|---|
|50|+1.998440|+0.641560|+0.382313|
|100|+1.159728|+0.373860|+0.136142|

All four root-period incremental gains are positive; all four forecast gains
exceed0.5. Development gate met; freeze the third update for Stage133. All four
refresh fits choose scale1. Root410011 period100 calibration slightly favors
radial continuation (0.386471 vs0.365270), including negative refresh-minus-radial
A-to-B crossfit; retain this rather than choose an evaluation-winning method.
These are equal-development-root means, not confirmation confidence intervals.

## Limitations

This adds a third training step and simulation cost. Radial continuation shares
the two-step learned source; it is not a fully stale three-step training route.
Two development roots do not confirm generalization. Per-update radius matches,
not cumulative distance from zero. Joint-HRL, promotion and frequency claims
remain unchanged; the Stage131 gate and negative root-level results are retained.
