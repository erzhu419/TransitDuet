# V54: road/time-conditioned mean persistence

Keep V48/V50 inputs and the real U200/L200 controller frozen. The V53 global
two-coefficient fit is the parent; fit road-specific coefficients, then road /
time-context coefficients. Time context contains the issued origin's quarter
hour and whether the queried departure crosses its hour boundary. Both are
calendar features, not future traffic observations. Smooth horizon interpolation
and [0,1] coefficient bounds retain a convex V48/V50 mean mixture.

Each local fit has ten parent-centred pseudo-pairs, scaled to its observed
feature leverage; sparse contexts shrink to roads and sparse roads to the
global fit. Only completed legs with end_s strictly before issue time and
previous predictions issued strictly before departure enter the three-hour /
4096-pair history. Future completions are evaluation labels only.

Run scheduler smoke, 12-day preparation and the unchanged V53 gate: every
cohort's 0-900 s MAE must be no worse than V50 and 900-1800 s no worse than V48.
No tolerance, cohort exclusion or retrospective gate tuning. Only a passing
prefix model proceeds to a 72-origin closed-loop screen with holding, passenger
and noise processes unchanged. All six demand seeds remain development data.

## Results
Smoke t160188 passes 16 tests. Preparation t160189-t160193 completes all 12 days;
gate t160195 fails: short-lead MAE delta versus V50 -1.345 s [-1.725,-1.021],
long-lead delta versus V48 +1.163 s [0.947,1.361], six demand clusters. The
long-lead gap narrows from V53's +2.019 s, but every cohort still fails. No
closed-loop matrix or real-policy promotion is launched. The short-lead gain
is not full forecast qualification.

Read-only audit t160216 finds same-sign V48/V50 mean biases in 81.25% of late
day/road/time cells (descriptive cell averages, not event-weighted inference).
For False:10 / 0:0, their biases are -17.40/-29.01 s and V54 remains -26.52 s.
This motivates testing completed-prequential conditional residual bias alongside
persistence next; the cell averages alone do not establish a query-level bound.
