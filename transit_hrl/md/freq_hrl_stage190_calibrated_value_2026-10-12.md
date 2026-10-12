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

## Limitations
This remains a development single-action value screen with greedy continuation,
not repeated learned-upper deployment or end-to-end HRL. Empirical margins do
not guarantee coverage under regime/lower-policy shift. The four critics use
less fitting data and more optimizer work than Stage188; this is not a
matched-compute architectural superiority experiment. Third/fourth reserved
scene action labels remain closed. Results are pending.
