# V103: held-future local-return teacher

V102 completes all 20 native replay/refit items and aggregation, retry 0.
Only its 59822-byte summary is retrieved. Cross-day combined action-difference
MSE: untouched 0.07166, TD 0.09209, MC 0.10255. MC-minus-untouched MSE interval
crosses zero, but combined MAE rises +0.04864 [0.00372,0.09279]. Both fixed-
budget repairs fail; do not extend their updates or distill their forecasts.

MC predicts fixed-charge soft gain +0.01421, measures -0.00264
[-0.02219,0.02146]. Native soft gain is -0.00229 [-0.01987,0.02043]. Full-day
passenger/CV/readiness effects remain unresolved. This is not evidence that
MC directly causes system harm; retain the separate V99 objective conflict.

Before another critic/control redesign, ask whether better conditional local
values would help under the unchanged objective. Use V101's cached seven-action
labels: estimate reward/cost on futures 0-1, compute the same soft optimum with
original alpha/dual/uncertainty charge, evaluate on futures 2-3, then swap.
Compare with actor and original critic law on exactly those held labels.
No coefficient, cutoff, gate or action is selected by evaluation labels.
Report crossed actor/day intervals and seed/day/time axes. Also review V102's
gain calibration and local/system sign conflicts by those axes. All computation
and tests use scheduler on node001-node006; no new rollout or optimizer step.

Decision: held-future local gains with replicated physical harms strengthen
the case for system-credit repair; joint benefits justify investigating direct
action-difference supervision. Unresolved local gains do not establish either.
This teacher is a diagnostic, not a deployable controller or a perfect-value
upper bound. Both consumed days and all four futures are development evidence;
no fresh-day or complete-policy claim. Main, actors, upper and dual stay fixed.
