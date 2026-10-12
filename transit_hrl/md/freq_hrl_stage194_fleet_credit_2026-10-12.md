# Stage194: Reference Risk and Action Credit

Keep600 states,30 mean8 targets and17 grouped folds; freeze nonfleet/wait.
Cross shared versus separated credit with absolute versus pair-RMS loss units.
Reuse the shared/absolute control and fit the other three versions. All have
1601 parameters,100 AdamW updates and the same16 initial hidden rows. Separated
uses disjoint8+8 networks: original-action risk anchor and15-pair credit loss,
with reference risk detached from pair gradients and zero contrast-output bias.
Restore physical units and the historical fleet floor before action selection.
The preregistration records exact losses, scales, grouping and budget.

## Result
Run `native_transit_fleet_credit_stage194_fit_20261012_r1`, source `5b626463fd`.
Fits `t145316-t145332` and report `t145333` completed exit0. Seven tests pass;
cached predictions reproduce Stage193. Budget:51 new fits/5100 updates,34
reused fits, no native episodes/actor updates. Summed fit wall time is57.3s.
All variants fail the joint cost/wait screen.

Against the matched fleet-only control, primary scene cost changes+0.000879
to-0.001470 and wait+0.004167 to-0.000208 minutes. Regime/root397 still has
cost+0.000307 and wait+0.002092; root401 wait is+0.027117. The stronger shared
three-head control has better scene cost. Pair MSE improves on regime but
worsens on scene/lower-root holdout; separate-model loss scaling changes little.
One226,254-byte summary was pulled; full data and models remain server-only.

## Next Step
Broaden conditional means across complete scene groups and all six callbacks,
balanced across roots/regimes. Refit nonfleet/wait alongside fleet credit:
primary still makes10-13 falsely feasible choices per30-state split.

## Limitations
The same ten development prefixes are reused and only5% of targets are means.
No full native learned-HRL or general-superiority claim is established; the
reserved-label/deployment gates remain closed.
