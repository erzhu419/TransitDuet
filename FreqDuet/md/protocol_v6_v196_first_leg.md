# V196: preregistered first-road half-change decomposition

Original64 days2951001-2951064 only,V194 saved moments plus V191 support and
V195 result references. No trajectory rescan,interval/predictor fitting,new
confirmation,replay,training or policy change. Freeze V151,V146,V172,3600s,
.1/.1 tails,actors/critic/dual and main. Do not retune RMS thresholds.

Before new readout,register three diagnostic questions:does the half change
already exist in raw covariance;does fixed-control adjustment create or cancel
it;does it concentrate in individual actors/common road keys?

For first-road underprediction Y and unabsorbed signed news J,use the SAME
V19524-control operator from average-half WITHIN-day covariance. Split:
adjusted=raw-cov(fittedJ,Y)-cov(J,fittedY)+cov(fittedJ,fittedY).
Projection is the sum of the last three terms. Delta is last32-first32 for
every term. Symmetrically allocate cross-control products to fixed clock/
geometry,forecast/bounds,leg-support,completed-service,stage and RMS blocks.
No per-half/actor/key refit,control selection or sequential causal attribution.

Keep V191 common actor/road/entry-hour keys,19 complete-input days in EACH half,
fixed support-only mass balance and original nominal query/episode weights.
Within-day is primary;between-day/total use that same within-day operator and
remain separate. Reproduce V195 points AND bootstrap95% intervals before
interpreting new attribution.10000 crossed actor/day draws,seed20260803;
shared actors,independent half days,fixed masks/balance,recomputed common
operator per draw. Eight primary delta contrasts:two modes x clearance/hour13
moving-launch x raw/projection;two-sided Bonferroni alpha=.05/8. All remaining
intervals are diagnostic95%,including overlapping hour13 and adjusted delta.
Point sign changes are not established opposite population signs.

Actor and actor/common-road-key contributions retain weights and the common
operator;their sums must reproduce each global term. Pool matching road labels
only for presentation,not support. Save all key/actor allocations server-side;
actor intervals are descriptive95%. Display largest3 absolute key deltas only
for readability,no selected-key tests or signed percentage shares. Subtract
each group without refitting/renormalizing to show point concentration;this is
accounting sensitivity,not a simulated counterfactual. No V195 gate reopening.

Six server mode/scope workers after algebra/weight/bootstrap tests;high priority,
explicit manual scheduler dispatch,node004. Local retrieval:small summary only.
Next decision must separate raw and projection changes from evidence of a
stable mechanism. Calibration associations do not establish causality,nonlinear
state-composition changes,held-future skill or controller improvement.

## Result and next decision

t210896/sourcef6c09a6dd1 completed high/manual on node004,exit0,six workers.
Six tests passed;all six cases reproduced V195 points AND bootstrap intervals.
Actor/key contributions and bootstrap actor sums reproduced the global terms.
Only278193-byte summary fetched:freqduet/results_freqduet/protocol_v6_v196_first_leg/audit_f6c09a6dd1/summary/status.json.

Within-day point covariances(squared seconds,original nominal mass):

| Mode/scope | Raw first32 -> last32 | Projection first32 -> last32 | Adjusted delta |
| --- | --- | --- | --- |
| old/clearance | -17.143 -> -7.942 | 14.577 -> 11.164 | 5.789 |
| handoff/clearance | -17.323 -> -7.695 | 14.609 -> 11.216 | 6.234 |
| old/hour13 moving-launch | -15.184 -> -7.627 | 12.967 -> 11.098 | 5.688 |
| handoff/hour13 moving-launch | -15.373 -> -7.533 | 12.944 -> 10.985 | 5.880 |

The raw points do not reverse sign. Positive projection offsets make the
adjusted last-half point positive,but the positive HALF CHANGE is already in
raw covariance;projection changes partly offset it. All eight raw/projection
delta family intervals cross zero. Only handoff/clearance adjusted delta has
positive diagnostic95% CI[0.259,12.012];this does not establish replication.
Forecast/bounds block delta is negative95% in clearance,both modes;other blocks
and moving-launch contrasts do not identify one stable controlling mechanism.

All12 actors contribute positive raw AND adjusted deltas in each primary case.
All30 pooled road keys have mixed signed contributions;the largest three are
(0,12,13),(0,11,13),(1,11,13),with order varying by case. Every one-actor and
one-road-key subtraction preserves the adjusted negative-to-positive point
pattern. Road leave-out adjusted deltas remain3.232-6.392 across primary cases.
Thus no single actor/key explains the point pattern;this is fixed-projection
accounting,not independent-day or deletion-counterfactual confirmation.

Between-day adjusted deltas are negative(-5.532 to -4.210),partly cancelling
within-day positive points;total adjusted deltas are only0.157-2.024. All of
their95% intervals cross zero. Keep these partitions separate.

Decision:do not fit a J correction,reopen V195 gates or launch confirmation.
If investigating further,preregister checks of observable-state covariance
and raw J/Y changes inside the existing common keys;do not tune RMS or remove
outcome-selected keys. Frozen items and main remain unchanged.
