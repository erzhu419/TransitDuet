# Stage124: Learn an Upper Step From Native Credit

Stage123 supplies reproducible native action derivatives at causal states.
Use its existing A/B labels to update the zero-mean upper actor; keep the
strong forecast teacher, lower residual, critics and policy variance frozen.

Reconstruct eight causal states per root/period with one zero rollout each.
Their suffix returns must reproduce the cached Stage123 zero paths. Fit the
direction of the pooled loss -mean(mu(state) dot native_return_gradient).
Take one Fisher-scaled mean-weight step and its opposite-sign control.
The radius is fixed at0.005^2/(2*0.15^2)=0.000555556, matching the earlier
action-vector scale. No evaluation-based radius or checkpoint selection.

Test32 new full1200-step scenarios per root at both periods50/100, paired
across source-flat, source-forecast, native-ascent, native-descent and
upper-blinded. Learned upper runs at every scheduled decision, not only at
the training query. Blinded must exactly reproduce forecast. The lower sees
forecast advice plus the same bounded0.05 donor-response channel.

New work per root:16 state-replay episodes +320 evaluation episodes =
336episodes/403,200steps. Two roots total672episodes/806,400steps.
Inherited label cost is separate: Stage123 used1,216episodes/1,459,200steps.
Four upper-only candidate weights per root remain server-side; no native
traces or checkpoints are downloaded. Four workers+parent/8GiB per task,
dynamic scheduler placement across node001-node006.

Four tests passed: signed weight update/fixed radius, exact cache-state
replay, frozen lower/common-noise/blinded closed loop and measured budgets,
fresh seed rosters with unchanged0.5 practical-gain threshold.

## Evidence Boundary

This is a two-root development pilot, not independent confirmation or joint
HRL training. Native finite differences require environment queries and a
strong warm-start lower; their cost is not comparable to free PPO labels.
Success requires fresh full-episode gain, not training-surrogate alignment.
Stage67 HOLD and the Stage121 negative result remain unchanged. Promotion,
strict LF/HF responsibility separation and cross-domain proof remain open.

## Run Receipt

Code `0c85dcfee0`; run `pointmaze_native_upper_step_stage124_pilot_20261006_r1`.
`t135891` root410011 and `t135892` root410023 accepted, initially queued.
Commands, cache identity, fixed radius, fresh scenarios and separate inherited
cost are saved in `preregistration.json`. No joint-training extension submitted.

## Completed Result

Both tasks DONE (node004/node006); fetched21.8KB result JSON only.
Ascent-minus-forecast: root410011 p50=-0.0074, p100=+0.1796;
root410023 p50=-0.1788, p100=-0.0357. Equal-root means=-0.0931/+0.0719.
Positive opposite-sign effects remain at both periods. Their odd component
is+0.2263/+0.1479, while the even deployment component is-0.3194/-0.0760.
This is consistent with closed-loop curvature/interactions cancelling the
direction signal; it does not identify their unique cause. No gain gate closed.
Next: Stage125 estimates step size using new training-only full-policy returns.
Keep this evaluation unchanged; no retrospective radius choice or seed extension.
