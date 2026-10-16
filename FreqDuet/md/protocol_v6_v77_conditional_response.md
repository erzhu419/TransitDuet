# V77: bounded conditional-response development test

V76 attributes about 98% of validation passenger MSE to state-to-curve mapping,
not quadratic action shape. High-mass training prefixes dominate the frozen
optimistic score. Combine this with the V55-V75 review: test heterogeneity and
known continuation-policy mixing, not another data expansion or long training.

Use only V75's 44 training days, 528 prefixes and all 32 paired futures.
Exclude one complete demand day across all four actors/three windows per fold.
Refit every scaler, target RMS and model on the other 43 days. Do not read
`prepared_validation.npz`: those twelve consumed days are not confirmation.

Fixed 2x2 comparison: pooled/policy-conditioned by linear/nonlinear. Pooled
inputs are unchanged V75 original causal state plus 38 current resource fields;
conditioning adds four indicators identifying the known deployed V59 actor.
No demand-day ID, future outcome or realized return enters inputs. This tests
sharing among known continuation policies, not generalization to new actors.
Zero-difference and action-only ridge are additional controls. Linear fits use
V75's exact day-balanced ridge=1 objective. The two nonlinear models have one
shared width-8 tanh layer and ten outputs, two quadratic coefficients per metric;
native-action differences are exactly zero by construction. Fixed seed 77000,
2,048 full-batch Adam steps, LR .003, weight decay .001. Only fitting-day RMS
normalizes output units; all untrimmed paired returns enter squared loss.
The nonlinear pair has the same update budget; ridge is closed form, not
compute-matched to Adam. Save fit loss and in-sample choice diagnostics.

Primary is policy-conditioned nonlinear, fixed before any V77 fold result.
Preserve the native-action joint gate: passenger effect CI upper <0, all four
constraint CI uppers <=0. Also require selected passenger effects to improve
over same-fold pooled linear (paired CI upper <0) for this development stage.
Report factorial contrasts, MAE/MSE/sign/order, predicted versus actual selected
effects, long/short choices, both fixed 16-future halves, actor/window effects
and demand-day gain concentration. Average contexts within day for intervals.
Prediction errors are diagnostics, not a requirement to strictly beat an
identical control. V75's failed gates and verdict remain unchanged.

Server smoke: five focused tests and the first full real fold; reuse that fold
in the formal matrix. Run the remaining whole-day folds in scheduler CPU shards
on node001-node006, then aggregate there. Pull only small markers/summary JSON.
No new rollouts, actor/dual updates, production-value changes or main switch.

If the fixed primary passes, next confirm single-intervention choices on fresh
days before a few-decision closed loop; do not add single-intervention gains.
If it fails, stop expanding this selector family and return to system-credit
regularity-constraint learning and physical holding permissions. Do not rescue
it by clipping labels, retuning margins, selecting another factorial cell or
claiming insufficient history has been established.

Limitations: development labels were already inspected; overlapping LODO fits
and only two original training seeds limit inference. A negative width-8 model
does not prove that current observations lack all usable information. None of
these development comparisons admits a deployable controller.

## Execution

Source `b6e3e10964`. Scheduler smoke `t166100` completed on node004, exit 0,
five tests and the full 700001-held-day fold passed. CPU batch planner assigned
four 11-worker shards `t166101-t166104` to preferred node002/004/005/006;
all remain eligible only for node001-node006. Aggregate `t166105` waits for
their four small completion markers. Reuse the smoke fold, add no rollouts,
and retain checkpoints/predictions/labels on the server.
