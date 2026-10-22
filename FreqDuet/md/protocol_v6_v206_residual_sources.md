# V206: residual uncertainty sources

Original64 saved moments and V205 draws only;two fixed keys,old/handoff and
clearance/hour13_moving_launch. V199 masks/bins/19-day gate,V191 factors,
original denominators,states and full parent/day centering remain fixed.
Reproduce all V205 residual part/mix/endpoint draws and their shared operators.

V205's refitted operator depends on BOTH actor and day weights;V201's analytic
linear-actor variance formula does not apply. With point weights A0,D0,split
F(A,D)-F00 into F(A,D0)-F00,F(A0,D)-F00,and the remaining joint response.
Use the SAME saved10000 actor/day draw pairs. These anchored responses add
exactly;their signed covariance allocations,including negatives,are NOT an
orthogonal ANOVA. Actor-only/day-only sensitivities refit the unchanged operator.

Separately partition within-cell/day,between-day and centered-mean-product
residual terms;retain between-term and raw-cell/parent-center cancellation.
Allocate actor contributions under the SAME saved joint operator,never an
actor-specific fit. These views overlap;do not add their shares together.

Delete each of64 SHARED days(1/31 in that half,other half unchanged) and each
of12 actors(1/11),refitting the same operator with ORIGINAL masks/factors.
Top3 shifts/allocations are localization only,not exclusions or new gates.
Unsupported exposure keeps only original aggregate JY/mass,no residual split.

Six server tests check saved/shared operators,nonlinear anchor additivity,
signed cancellation,temporal parts and delete-one weights. High/manual on
node004;only small status JSON local,no raw trajectory scan,confirmation,
interval/predictor fit,replay,training or frozen-controller change.

Limitations:reused development data and bootstrap sensitivity,not causal
uncertainty. Day-mean covariance is not day-resampling variance. No ETA or policy
improvement claim;no reason to discard influential actors/days.
