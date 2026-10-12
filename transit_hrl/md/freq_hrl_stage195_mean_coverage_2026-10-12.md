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
`e24921e689`, ids`t145379-t145388`. Measured peak RSS690-715MB permits1536MB
for subsequent native collection; offline tasks retain3072MB. Input/label
contracts and all fitting budgets are unchanged by this resource adjustment.
Full context, draw labels, native rollouts, target map and models stay remote.

## Next Step
If broader targets reduce false wait-feasibility and pass the joint screen,
test the frozen learned selector in repeated native upper decisions. Otherwise
use the complete-group wait/cost diagnostics before changing the objective.

## Limitations
These are forty existing development prefixes, not240 independent scenes or
new held-out confirmation. Mean8 retains sampling noise and only resamples
future innovations, not latent profiles. No end-to-end learned-HRL claim.
