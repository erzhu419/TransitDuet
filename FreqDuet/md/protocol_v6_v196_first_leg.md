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
