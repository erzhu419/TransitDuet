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
third-scene frozen transfer.

All twenty source baseline peaks match the registered first/second scenes;
427/720 future fleet-cost targets are zero, range [0, 2/3]. Twenty-six
focused tests pass, including absolute-label construction, physical units,
floor/identity, wait selection, held-out target isolation, shared fit budgets
and previous value-model behavior. Actual fits run only on the scheduler.

Run native_transit_floor_value_stage181_fit_20261011_r1, source eecc5aa084.
t143187 raw_absolute ran on node006; t143188 floor_absolute on node004.
Both completed exit0: 56 fits/84,000 updates qualified. The two compact
results (3,322,468 bytes) were already local; no additional pull was needed.

## Result and Next Step (2026-10-12)

Primary floor_absolute:wait_feasible still fails both registered gates:

| Split | root397 cost | root401 cost | Overall wait, min |
|---|---:|---:|---:|
| Scene | +0.0082361 | -0.0051346 | +0.0196583 |
| Regime | +0.0006878 | -0.0126052 | +0.0095250 |
| Lower root | +0.0018553 | -0.0251197 | +0.0247333 |

The floor removes all invalid absolute predictions (raw has 226 scene,
243 regime and 222 lower-root violations). It improves lower-root costs,
but not scene costs consistently. Wait selection also has mixed effects;
all six root/split wait means remain positive. No variant passes both gates.

Scene primary predicts wait -0.154325 but realizes +0.019658. Across all
nonidentity candidates the mean wait residual is only +0.006129; among
changed selected actions it is +0.267667, with 47/78 falsely feasible.
Apparent training cost/wait improve (-0.028910/-0.034425), unlike transfer.
This supports testing overfit/selection optimism, not a global mean offset.
Next: nested, training-only scene checkpoint selection and full training-fold
refit, compared with the existing fixed1,500-step floor model. Do not reuse
outer labels to choose stopping or open third/fourth action-label scenes.

## Limitations

A valid absolute floor is not a guarantee that relative savings or wait
predictions are calibrated. Feasibility is predicted, not assured in the
simulator. These are development one-action labels under original greedy
continuation, not repeated learned-action control or end-to-end HRL proof.
Third/fourth registered scenes' action labels remain unused.
