# Stage195: Complete-Group Conditional Means

Expand mean8 targets from30 to240 of the unchanged600 development states:
Stage187 replicas0-3, both lower roots, all five regimes, all six callbacks.
Reuse all240 Stage191 draws for30 states; collect1680 new conditional draws
for210 states. Eight frozen native episodes per new draw give13440 episodes.
Only future Poisson/speed innovations change after the exact original prefix;
latent demand profile, physical state, lower weights and greedy suffix remain.

Refit fleet, nonfleet and wait together with the unchanged scalar three-head
critic:1635 parameters,100 AdamW updates. Reuse47 Stage188 single-label models
with identical input normalization, scene groups and fitting budget. Forty
scene folds, five regime folds and two lower-root folds exclude whole groups.
Evaluate both models on the same240 states and same mean8 outcomes. Require
negative physical cost and nonpositive wait for both roots in all three splits.

Scheduler phases: ten replica1/probe9/draw0 qualification tasks, remaining1670,
remote target preparation,47 fits, compact report. Nodes001-006 are unbound.
Ten qualification tasks completed exit0 using one CPU/3072MB, source
`e24921e689`, ids`t145379-t145388`. Measured peak RSS690-715MB permits a1536MB
request for subsequent native collection; offline requests retain3072MB. Input/label
contracts and all fitting budgets are unchanged by this resource adjustment.
Full context, draw labels, native rollouts, target map and models stay remote.

Remaining1670 tasks registered as`t145389-t147058`, source`d0890febf4`.
Scheduler adjusts live RAM estimates from observed family usage (816MB at
release); `remaining_release.json` separates requested from scheduled memory.
All1680 collection tasks are now done/exit0. All1680 remote receipts qualify,
covering13440 new native episodes; peak process RSS is733.6MB. Preparation,
47 fits and the performance report are pending.

Scheduler completion accounting stalled after the processes exited: node-batch
timeouts discarded completed chunks, and exit0 without a recognized text marker
remained ambiguous. Fix`7bef2eb` preserves chunks, accepts confirmed clean exit0,
clears dead-task live usage and retains backend finish times. Only the control
service was restarted. Terminal regression:79 passed; two existing ETA-format
tests remain mismatched. No raw artifacts were downloaded for this diagnosis.

## Next Step
If broader targets reduce false wait-feasibility and pass the joint screen,
test the frozen learned selector in repeated native upper decisions. Otherwise
use the complete-group wait/cost diagnostics before changing the objective.

## Limitations
These are forty existing development prefixes, not240 independent scenes or
new held-out confirmation. Mean8 retains sampling noise and only resamples
future innovations, not latent profiles. No end-to-end learned-HRL claim.
