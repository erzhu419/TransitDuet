# Stage158: Frozen Plan Credit Diagnosis

Stage157 learned plans execute, but physical cost/wait do not improve
consistently. Both apparent cost gains over constant residual are dominated by
one scene's peak-fleet step. Do not expand training before identifying why.

Freeze both final Stage157 actors, critics and native lowers. For each root
and each of five regimes, use the first registered scene, without selection.
Reproduce learned, forecast and constant-residual episodes exactly. Then at
macro decisions 8, 22 and 36, replace ONLY that action with zero or its negative;
continue the same frozen deterministic actor afterwards. Require exact matching
of all causal observations/actions through the pre-intervention state and the
pre-action cost. Fixed scene, service endpoints, fleet and episode clock remain.

Measure the true terminal physical-cost/wait effect against the original learned
episode and compare its sign with the frozen twin-critic action margin at the
SAME state. Record action saturation and per-feature state ranges. This separates
executable action authority from a critic's preference for boundary actions.

Also subtract forecast prefix credit at identical macro decision clocks.
The paired sum equals `100*(forecast_final_cost-learned_final_cost)` exactly;
compare its scale with the raw prefix-cost credit. This diagnoses common-mode
service-progress variance without changing the learner or its objective.

Ten scheduler tasks, one root/regime each, 9 frozen episodes/task, 90 total,
5,524,200 native ticks, zero training. node001-006 unpinned, 1 CPU/3 GB each.
Only compact JSON is returned; existing server checkpoints stay on servers.
92 focused tests pass, including single-action replacement, exact prefix checks,
paired terminal credit, unrounded physical effects, analyzer rejection of changed
interventions/budgets and code-only unpinned scheduler placement.

## Limitations

One diagnostic scene/root/regime, not a new performance or statistical claim.
The critic estimates a soft stochastic continuation while the deployed actor
is deterministic. Ranking disagreement is a deployment diagnostic, not by
itself a Bellman-error proof. A paired-credit variance reduction on these paths
does not prove that a counterfactual-baseline learner will improve performance.
These interventions test the existing within-window action representation;
they cannot establish cross-window service allocation or full joint HRL.
