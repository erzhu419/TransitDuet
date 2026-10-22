# V197: common-key observable states and raw first-road covariance

Original64 days2951001-2951064,V194 saved moments only;V191 support and V196
readout references. Freeze V151,V146,V172,3600s,.1/.1 tails,actors/critic/dual
and main. No trajectory rescan,interval/predictor fit,confirmation,replay or
training. Keep exact common actor/road/entry-hour keys,19 complete-input days
in EACH half,original nominal query/episode weights and V191 support balance.

Preregister:check means/variances of ALL24 inherited observable controls plus
J and first-road error Y;within-day covariance/correlation with J and Y;and
fixed-projection state/mixed/residual accounting of the raw covariance change.
Report state moments per retained mass;primary covariance retains nominal mass.
Scale mean deltas by average-half within-key SD. Zero-variance correlations
and standardizations are unknown. No control selection,threshold or new bins.

Use the SAME V195 pooled within-day operator in both halves and all keys:
J=fJ+rJ,Y=fY+rY;raw=cov(fJ,fY)+cov(rJ,fY)+cov(fJ,rY)+cov(rJ,rY).
Mixed is the sum of the cross terms. Allocate state-state and mixed terms to
the fixed V196 control blocks,including all cross-block products. Do not refit
by half/key. Pooled mixed cancellation is a projection identity,not validation.

Means are descriptive:within-key/day means were already removed from primary
covariance. Balanced point key masses preserve the key mixture;bootstrap means
use resampled masses without a second renormalization. Report between-day and
total separately. Second moments cannot establish nonlinear conditional mix,
causal state effects or a state-matched counterfactual population.

10000 crossed draws,seed20260803,shared actors/independent half days;fixed masks
and support balance,recomputed common operator. Eight within-day delta tests:
two modes x two primary scopes x(state covariance,mixed),Bonferroni alpha=.05/8.
Other intervals diagnostic95%;V195 gates unchanged. Reproduce V196 points AND
bootstrap intervals before interpretation. Hour13 overlaps clearance.

Save ALL pooled-key state points and actor/key term allocations server-side.
Report counts of positive/negative/zero/unknown key deltas,not independent
replications. V196-selected keys(0,12,13),(0,11,13),(1,11,13) are development
diagnostics only;never drop other keys or claim fresh confirmation.
Six mode/scope workers,server tests first;high/manual scheduler,node004.
Only the compact global summary is retrieved locally. No J correction or
policy promotion follows from this diagnostic alone.
