# Stage190: Training-Only Cost/Wait Calibration

Stage189 found selected-action optimism despite small all-candidate bias. An
oracle wait mask still left positive regime/root397 cost, so wait gating alone
does not resolve the failure. This experiment corrects both cost and wait risk
without collecting more native episodes.

The fixed Stage188 bank remains server-only: 600 states, 100 independent scene
groups, 3,600 candidate labels. Both scalar and timetable-sequence critics use
the same 107 outer folds. Within each outer training set, a seed190 root/regime
stratified partition reserves 20% of scene groups for calibration. Four critics
each omit a quarter of the remaining proper-training groups, retain the original
100-step proper-MSE optimizer and seed178, and normalize only their own inputs.

Uncertainty is the across-member standard deviation of paired total cost or
wait, floored at 10% of proper-training target RMS. Calibration uses the maximum
standardized underprediction across five nonidentity actions and six callbacks
per scene. The fixed 90% upper-rank multiplier is estimated separately for cost
and wait. Selection minimizes upper cost subject to nonpositive upper wait;
identity is exactly zero and wins ties. No outer outcomes set these margins.

Compare ensemble point selection, calibrated selection, and cached Stage188
selection on identical held-out states, including the original 120-state subset.
Report fallback rate, selected optimism, bound violations, cost and wait by
root/split. Zero-cost fallback alone cannot pass the strict improvement gate.
Budget: 214 scheduler fit tasks, 856 critics, 85,600 updates; zero new native
episodes or actor updates. Nodes001-006 are eligible without node binding, with
one CPU and 3GB per task. Only compact receipts/report sync locally.

Run: `native_transit_calibrated_value_stage190_fit_20261012_r1`, fit-source
revision `4213e1d860`, scheduler tasks `t144436` through `t144649` (214 tasks).
Report task `t144650` qualified all 856 fits and 85,600 updates. One manual
pull fetched only the 56,023-byte report; bank/models/detailed predictions stay
remote. Eight focused synthetic/mock tests pass.

## Result and Next Step
Both calibrated representations fall back on all 600 states in each of three
splits: cost and wait are exactly zero, so strict cost improvement fails. Do
not count zero selected bias or zero bound violations as learned control gains.
Ensemble point selection improves some cells but still fails the joint gate:
scalar regime/root397 cost is +0.000599; sequence regime/root397 is +0.002527.
Both retain positive wait in several root401 splits. Sequence fitting takes
11,134 seconds summed across tasks versus 379 for scalar, about 29 times more.

This rules out this conservative selector as the current improvement route,
not all causal plan learning: a scene-wide worst-action margin is substantially
stricter than the original mean cost/wait gate. Do not retune its quantile on
these test outcomes. Next investigate whether single-continuation labels expose
predictable action differences, then joint active-bus/plan representation and
expected continuation-value targets rather than another gate/seed sweep.

## Limitations
This remains a development single-action value screen with greedy continuation,
not repeated learned-upper deployment or end-to-end HRL. Empirical margins do
not guarantee coverage under regime/lower-policy shift. The four critics use
less fitting data and more optimizer work than Stage188; this is not a
matched-compute architectural superiority experiment. Third/fourth reserved
scene action labels remain closed. No reserved-screen/deployment gate passes.
