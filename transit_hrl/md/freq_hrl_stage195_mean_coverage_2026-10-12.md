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
All1680 collection tasks/receipts qualify, covering13440 new native episodes;
peak RSS733.6MB. Preparation`t147113`, fits`t147123-t147169` and report`t147179`
completed exit0, source`0aaf516d48`. Budget47 fits/4700 updates plus47 reused
controls; summed fit wall103.2s. One506,559-byte report was manually pulled;
the589,204-byte target map, full feature bank and models remain server-only.

Scheduler completion accounting was repaired by`7bef2eb`; backend finish times
and completed native evidence are preserved in the collection receipt.

## Result
Primary mean8/wait-feasible: scene cost-0.001373/wait-0.001371min; regime
cost+0.001356/wait+0.000588min; lower-root cost-0.002459/wait-0.004965min.
Lower-root improves both roots versus the matched control, but the joint screen
still fails: scene/root397 wait+0.001705min, scene/root401 cost+0.0000447, and
regime cost is positive for both roots. False feasible choices are76/78/67 for
scene/regime/lower-root, versus68/70/67 for the control. No general win.

## Next Step
Stage196: reuse all eight draws and frozen47 fits to separate sampling noise
from fleet/nonfleet/wait errors and check odd/even choice transfer. No further
native collection or fitting before identifying the source of the failed gate.

## Limitations
These are forty existing development prefixes, not240 independent scenes or
new held-out confirmation. Mean8 retains sampling noise and only resamples
future innovations, not latent profiles. No end-to-end learned-HRL claim.
