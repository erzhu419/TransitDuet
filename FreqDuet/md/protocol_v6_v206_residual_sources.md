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

## Result and next decision

t210963/source3db715e6c0 completed high/manual on node004,exit0;6 tests passed.
V205 residual points/parts/mix/endpoints/operators reproduce(max1.66e-10).
Only127389-byte summary fetched;all actor/day arrays remain server-side.
Target is V205's conditional half-difference,residual-residual part,in weighted s^2.

Across8 cases,the point-anchored signed allocations are:day94.8-97.6%,actor
.4-2.7%,remaining joint1.5-3.5%. Day-only SD1.36-2.23 versus actor-only.078-.388.
Temporal allocation is separate:between-day47.8-66.2%,within-day13.7-35.9%,
centered mean-product11.7-23.2%. These are overlapping views,not additive shares.
For moving-launch/key7,raw-cell allocation803-1043% is offset by parent-centering
-703 to-943%;do not add own variances as independent noise.

Most sensitive days repeat across modes:key7 clearance2951020/2951038/2951051;
key7 moving-launch2951053/2951045/2951036;key12 clearance2951044/2951009/2951021;
key12 moving-launch2951033/2951009/2951021. Single-day maximum shifts across cases
are.473-1.023 weighted s^2;single-actor shifts never exceed.195. Deleting2951020
reverses key7 clearance's residual point sign;this is sensitivity,not an exclusion.

Decision:do not add actors or tune an interval/state correction on these64 days.
Any further mechanism work should target shared day-level residual co-movement,
not identify a bad actor or remove influential days. No new replicated mechanism
or policy gain follows;retain all frozen items and unsupported exposure separately.
