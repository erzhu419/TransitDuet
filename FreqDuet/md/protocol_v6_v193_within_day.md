# V193: within-day innovation variance,upper-event concentration and directions

Original64 calibration days only,V190 moments and exact V191/V192 common keys:
19 complete-input days in EACH half,same support-only balanced masses. Reproduce
V192 reference projection,analysis mass and within-day covariance contribution.
Focus clearance,hour13 and hour13 moving-launch,both modes. Retain all days.

Measure raw and reference-residual innovation variance within actor/key/day,
its fraction of total within-key variance,and raw/adjusted within-day correlation.
Compare upper-event versus exposure mass concentration across the32 days:
effective mass days=(sum mass)^2/sum(mass^2),plus largest-day share. These are
weight-concentration summaries,not independent event counts or statistical ESS.
Describe mass and innovation variance in actor/key/day cells with both binary
labels. Outcome-defined partitions are diagnostics,never deployment conditions.

Report each day's within-key raw and shared-reference adjusted direction;
undefined variance stays unknown. No per-day hypothesis tests or day selection.
10000 shared-actor/independent-half-day bootstrap draws,seed20260803;recompute
the V192 common reference residualizer each draw. The bootstrap second moment
uses sum(n_i*mass_i^2),not sum((n_i*mass_i)^2);each occurrence is a position. Keep
19-day masks and balancing fixed. Direct half differences,diagnostic95% intervals;
original V190 replication and family gates are unchanged.

Freeze V151,V146,V172,3600s window,.1/.1 tails and main. No interval/predictor fit,
new confirmation,replay or training. Tests and six analysis units run through
high/manual scheduler on node001-node006. Only small summaries/logs come local;
per-day and common-key detail stays remote. Binary moments do not identify
independent event/episode counts,and this linear diagnosis cannot prove causal drift.

## Results and decision

t210886/source24f4987b0b,high/manual node004,exit0,no retries. Six tests passed;
all6 cases reproduced V192 projection,mass and within-day contribution. Fetched
97338-byte summary only,no raw archives. Frozen items and main remain unchanged.

Old clearance(first/last32):raw within-day innovation variance77.26/94.51 s^2;
reference-residual variance32.04/35.84 s^2,42.1/43.7% of total within-key residual
variance. Its change is3.79[-4.40,11.57] s^2:no collapsed first-half variation.
Upper-event rate9.50/8.01%;effective event-mass days23.18/24.12 versus exposure
days31.96/31.96,largest-day shares8.38/6.89%. Event-mass effective-day change
is0.94[-3.39,5.08]. First-half evidence is not simply missing event mass or a
handful of dates. Cells with both labels carry25.8/22.6% of analysis mass and
42.6/49.6% of innovation variance;their differences remain uncertain.

Reference-adjusted daily signs are16 positive/16 negative then20/12(old);handoff
is16/16 then19/13 in clearance. Within-day r=-0.0037[-0.0689,0.0640] then
0.0335[-0.0255,0.0908],change0.0372[-0.0532,0.1229]. Hour13 agrees;moving-launch
r=0.0060[-0.0691,0.0851] then0.0408[-0.0146,0.0964]. Both halves remain weak
within-day,not an established first-to-last relationship change.

Raw daily direction is undefined on one day in each half. Adjusted residual
direction can be finite there through the control projection;it is not evidence
of an observed within-day upper-event contrast. These are descriptive directions,
not per-day discoveries. Weighted moments do not establish independent event counts.

Decision:do not promote this RMS upper-tail correction or tune its strata/window.
Preserve the existing cross-day association and failed replication separately;
neither establishes stable within-day risk skill. No interval fit,confirmation,
replay or training. A further candidate needs a new observable mechanism and
preregistration,not another adjustment to rescue these same associations.
