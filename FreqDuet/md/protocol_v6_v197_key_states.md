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

## Readout and decision

t210902/source87cad6a8a8 completed high/manual on node004,exit0,six workers.
Six tests and all six V196 point/bootstrap reproductions passed. Only966669-byte
summary fetched:freqduet/results_freqduet/protocol_v6_v197_key_states/audit_87cad6a8a8/summary/status.json.
Within-day nominal covariance deltas,last32-first32(squared seconds):

| Mode/scope | Raw | State covariance | Mixed | Residual |
| --- | --- | --- | --- | --- |
| old/clearance | 9.201 | 3.360 | 0.053 | 5.789 |
| handoff/clearance | 9.628 | 3.384 | 0.009 | 6.234 |
| old/hour13 moving-launch | 7.557 | 2.938 | -1.069 | 5.688 |
| handoff/hour13 moving-launch | 7.839 | 3.038 | -1.078 | 5.880 |

All eight state/mixed delta family intervals cross zero. The residual is the
part outside this fixed linear state projection,not identified physical drift.
All between-day core-term delta95% intervals also cross zero.

Diagnostic95%:Y correlations with first fraction,horizon and predicted running
weaken in all four primary cases. Example old/moving-launch:running0.203->0.087,
delta CI[-0.202,-0.029]. Horizon correlation decreases in20-23 of30 pooled keys;
running in21-23,first fraction in17-18. These are not independent replications.
Their mean shifts are not established;some other marginals DO change:clearance
predicted-entry/command shares fall approximately10.8->9.4%/8.4->6.8%,and service
prefix-age variance falls. Moving-launch has zero predicted-entry/command share
in both halves,yet the covariance pattern persists. Stage mixture alone is
therefore not a complete accounting explanation.

Decision:no J correction,state gate,interval fit or confirmation. Distinguishing
finer state mixing from within-state dependence would require conditional or
higher-order cross moments absent from these saved first/second moments;do not
infer that distinction from another linear decomposition. Any such extension
needs a separate preregistration;no collection/rescan is started here. All
frozen items and main remain unchanged.
