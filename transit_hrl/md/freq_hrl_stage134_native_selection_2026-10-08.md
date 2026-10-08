# Stage134: Select the Third Update by Native Validation

Stage133 clears the material forecast gate at both periods, but third-step
increment at50 remains inconclusive and refresh does not beat radial control.
The fixed-refresh rule ignores calibration evidence. Three-point quadratic
fits can also predict a positive scaled gain with negative native crossfits.

Return to development roots410011/410023 and the Stage130 two-step weights.
Keep the Stage132 third-update directions,26D projection,std0.15,epsilon0.005,
damping1,per-update radius0.000555556,authority0.05 and frozen lower/critics.
Four fresh label scenes/two panels and16 fresh step-calibration scenes/two
panels. Do not add a fourth update or switch methods by evaluation results.

Add16 separate native validation scenes/two panels. Execute the actual pooled-
fit refresh and radial candidates plus the unchanged two-step policy. Before
evaluation, choose the largest mean paired incremental native return; unchanged
policy has gain0 and wins zero ties. This selection does not use the quadratic
prediction or crossfit estimate. Then evaluate32 completely new scenes with
all seven existing controls. Selected-policy outcomes alias the already executed
chosen arm on the same paired scene; no extra eighth simulation is counted.

New cost/root:6,048episodes/7,257,600steps, including192validation episodes.
Four candidate checkpoints/root remain server-side; a selected checkpoint
reference also represents the no-update case. Two17CPU/12GiB dynamic tasks on
node001-node006; no waiting aggregation task; compact JSON only.

Development requires selected-minus-two_step>0 and selected-minus-forecast>0.5
at both periods. Selection-minus-refresh/radial is reported separately; if
selection does not help, retain that outcome rather than claim universal benefit.

## Run Receipt

Implementation `d89c486ad0`, registration `ac91cc4f19`;seven tests passed.
The two completed Stage130 source cells and four upper files are available on
node004; the native runner checks checkpoint protocol and source fits on load.
Run `pointmaze_native_selection_stage134_pilot_20261008_r1`:
`t136566` root410011 and `t136567` root410023 accepted; initial snapshot queued.
Exact task specifications, all four seed-role sets, budgets and compact scheduler
receipt are saved with the run. No Stage134 performance result is available yet.

## Limitations

This is training-time candidate selection, not an online learned promotion
gate or joint HRL proof. Validation incurs extra simulation cost and selecting
the best noisy mean can still overfit. Two development roots do not confirm
generalization; Stage133's negative results and composite HOLD are retained.
