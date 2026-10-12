# Stage201: frozen wait-selection diagnosis

Explain Stage200 before changing the wait critic. Reuse all47 prediction files,
240 mean8 states and all three margins; no fitting, inference or native episodes.
Decompose each policy into identity, retained, cancelled and rerouted actions.
Cost/wait changes are divided by the entire evaluation scope, not subgroup size.
Inspect all five nonidentity candidates' raw wait residuals by root, action,
queue (zero,0-50,over50 passengers) and prefix headway change versus identity.
The50-passenger threshold is the existing native feature unit, not outcome-tuned.

Run: `native_transit_wait_selection_stage201_diagnostic_20261013_r1`.
One scheduler CPU task,3072MB, unbound node001-node006. Detailed state records,
targets and cached predictions stay server-only; only the summary is pulled.
Task`t147592` completed on node005 with exit0;14 focused tests passed.
Only the144234-byte summary was pulled; no targets, models or state records.
The main failure is cancelling beneficial actions while retaining optimistic
wait predictions, not rerouting. In regime/root401, the primary cancels46
actions and reroutes none: cancellation accounts for the entire cost increase
(+0.000847921) and wait increase (+0.001844792 min) versus zero margin.
In scene/root397,15 retained primary actions include10 wait-harmful choices;
their predicted mean wait is -0.021632581 min but actual mean is +0.011750000.
Root397 scene selected wait residuals differ by queue: +0.004599489 min for
0-50 passengers versus +0.019622519 for over50 (association, not causal proof).

Next: freeze both cost heads and replace the wait predictor with conditional
action credit using queue/load and committed headway geometry. Fit all training
candidate labels, not only previously selected actions; retain all47 outer
folds, training-only normalization and the existing strict gate. No global
margin sweep or held-out-dependent threshold tuning.
Limits: descriptive reused-label diagnosis, not independent confirmation,
CI-supported improvement or full learned-HRL validation.
