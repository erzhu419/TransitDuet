# Stage181: Anchored Fleet Value and Learned Wait Feasibility

Stage180 improves some mean costs but fails both-root cost/wait transfer.
The paired loss leaves absolute fleet values unidentified; scene-selected
fleet savings are predicted as -0.185532 but realized as -0.006944.
Existing wait-feasible hindsight actions improve cost in 73/120 states.

Reuse all 120 states/720 labels/20 first-second scene groups. Recover the
exact original terminal fleet peak from qualified Stage169 source records;
this is a supervision label, never an inference feature. For each action,
future fleet cost = original terminal fleet cost + paired fleet delta -
cost of the already observed historical peak. It must be nonnegative.

Two matched 96-16-3 networks (1,603 parameters): raw_absolute and
floor_absolute. The latter predicts future fleet cost via softplus, then
adds the irreversible observed fleet floor. Both learn absolute fleet
MSE, all-three-head paired MSE and native-cost expected regret, with unit
loss weights and training-only scales. Fleet/nonfleet remain the only
cost terms; the wait head predicts minutes, not an extra cost weight.

Each fitted model supports cost_only and wait_feasible selection. The
latter minimizes predicted physical cost among actions with predicted
wait delta <= 0; identity remains exactly zero and wins exact ties.
Primary is fixed to floor_absolute:wait_feasible. Same seed178, AdamW,
1,500 steps, width16 and causal96 features as the previous screen.

20 leave-scene, five leave-regime, two leave-lower-root and one full fit
per method: 56 fits/84,000 critic updates, two unbound scheduler tasks on
node001-006, no new native episodes/actor updates. Both selectors share
weights and predictions. Primary must strictly improve cost and not
worsen wait for both roots in all three splits before separately registered
third-scene frozen transfer. Results pending.

All twenty source baseline peaks match the registered first/second scenes;
427/720 future fleet-cost targets are zero, range [0, 2/3]. Twenty-six
focused tests pass, including absolute-label construction, physical units,
floor/identity, wait selection, held-out target isolation, shared fit budgets
and previous value-model behavior. Actual fits run only on the scheduler.

Run native_transit_floor_value_stage181_fit_20261011_r1, source eecc5aa084.
t143187 raw_absolute runs on node006; t143188 floor_absolute on node004.
Both are unbound and startup logs confirm seven completed fits per task.
Completion, comparison and the fixed primary gate remain pending.

## Limitations

A valid absolute floor is not a guarantee that relative savings or wait
predictions are calibrated. Feasibility is predicted, not assured in the
simulator. These are development one-action labels under original greedy
continuation, not repeated learned-action control or end-to-end HRL proof.
Third/fourth registered scenes' action labels remain unused.
