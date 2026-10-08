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

## Limitations

This adds a third training step and simulation cost. Radial continuation shares
the two-step learned source; it is not a fully stale three-step training route.
Two development roots do not confirm generalization. Per-update radius matches,
not cumulative distance from zero. Joint-HRL, promotion and frequency claims
remain unchanged; the Stage131 gate and negative root-level results are retained.
