# Stage182: Nested Scene Checkpoint Selection

Stage181's floor is correct but its selected wait predictions fail transfer.
In scene cross-fits, mean wait residual is +0.006129 over all nonidentity
actions, but +0.267667 among changed selected actions (47/78 false-feasible).
Training cost/wait improve while held-out root397 and all wait means worsen.
Test whether fixed1,500-step fitting contributes to selection optimism.

Keep all 120 states/720 labels/20 scene groups, causal96 inputs, floor
96-16-3 network, losses, seed178, optimizer and cost/wait selector. Reuse the
qualified fixed1,500-step Stage181 floor model as control; do not rerun it.

Within each outer training fold, sort complete groups by scenario/root/seed
and hold every fifth group, starting at zero. Inner training-only scales;
fit for 1,500 steps and evaluate checkpoints 100/300/600/900/1200/1500.
Select minimum held-inner-scene absolute fleet MSE + three-head paired MSE,
earlier on ties. Validation never updates weights. Reinitialize with seed178
and refit all outer training scenes for the selected number of steps.
Outer scene/regime/lower-root labels never select or normalize checkpoints.

28 inner fits + 28 refits, 42,000 inner updates plus <=42,000 refit updates;
no control updates, native episodes or actor updates. One unbound scheduler
CPU task, node001-006 allowed. Both selectors reuse each refitted model.
Primary scene_stop:wait_feasible must improve cost and not increase wait
for both roots in all three outer splits before registered third-scene
transfer. Results pending.

Thirty-three focused tests pass: complete/outcome-independent inner groups,
validation-only monitoring, unchanged fixed-step weights/scales, minimum-loss
stopping, fresh full-fold refit, outer-target isolation, actual update budgets,
source-control pairing, fixed primary and previous critic behavior. Local
fits are shortened synthetic unit tests; research fitting uses the scheduler.

## Limitations

The method uses extra inner-fitting compute relative to the reused control;
this is not an equal-compute comparison. Only twenty independent scene
groups are available; inner checkpoint selection can itself be noisy.
It tests training-duration overfit, not all causes of selection bias.
One-action development labels do not establish repeated learned-action HRL.
Third/fourth registered action-label scenes remain unused.
