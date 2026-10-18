# V103: held-future local-return teacher

V102's complete fixed-budget TD/MC refits fail cross-day action-difference
validation; their native soft gains and physical effects remain unresolved.
Keep untouched critics and retain the separate V99 objective-conflict evidence.

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

## Outcome / Decision

Source `36bc64d3ac`, scheduler `t179166` on node004: four tests and complete
cached audit pass, zero new rollout/update. Retrieved only 150406 bytes.
Teacher native soft gain +0.04671 [0.01429,0.10731], versus original critic
+0.00245 [-0.00514,0.01187]; their paired difference +0.04426
[0.01129,0.10460]. All ten actor means, both days and both time means gain.
This is conditional local-value opportunity, not a learned observable policy.

Passenger exposure +705.53 [-1670.49,3622.91] person-s, CV squared +0.0000101
[-0.0000806,0.0001234], readiness +0.2011 [-0.2372,0.6202] s all remain
unresolved. Of 30 native-soft-positive contexts, 15 have higher passenger
cost, 13 higher CV and 19 higher readiness delay by point estimate; only six
are jointly nonworse. Morning passenger/CV/readiness means worsen; later means
improve. These signs neither prove replicated harms nor admit a time gate.

Next design priority is system-level action advantages for passenger time,
actual CV and readiness, separating return levels from action contrasts.
Account for V59's global-critic calibration failure, V75's decision errors
despite lower MAE, and V78's unresolved CV constraint. No new training matrix
is submitted; untouched critics/main remain active. Local-value accuracy and
system-credit alignment are separate requirements.
