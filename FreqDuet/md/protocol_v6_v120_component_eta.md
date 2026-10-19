# V120: prequential remaining-time components

Read the eight exact V118 funded paths only. Actors, critic, dual, quota and
main stay frozen. No new simulation or training. Scheduler on node001--node006
runs tests and analysis; only the final small summary returns.

Predict remaining road time, upstream service, holding, execution ticks and
target service separately. Use same-road/same-stop, same-direction labels
strictly completed before prediction, from the last 3600 s; when that window
is empty use the available completed history of the same key. No window or
shrinkage search. Service matures at ready, running at arrival, holding/ticks
at departure. Missing support keeps raw ETA, not zero components.
The first road starts at recorded physical launch, not an ordinary terminal
stop departure; every other road requires its adjacent completed departure.

Frozen comparisons: raw; replace running only; replace stop components only;
replace both. Components apply only when earlier arrival/departure events and
current AVL progress establish an in-transit follower. Pending/event-boundary
cases retain raw predictions. Validate launch-clock/route-coordinate agreement.
Record coverage and component error; compare ready MAE and same-object holding
signs, including corrected false positives and lost true benefits.

Advance only to independent predictor confirmation if split's paired crossed
95% upper bound is below zero for ready MAE and both cut cohorts' sign error,
with neither cohort's departure-sign error worse in mean. No control promotion
follows from this development-data diagnostic. Local signs are not system returns.

## Outcome and next decision

Source `a3822ea155`, scheduler `t189906`, node006: six tests, eight paths,
exit 0; 171,084 bytes returned. Earlier `t189904` was cancelled before analysis:
all 262 first roads per path start at physical launch, not a terminal stop event.
No intermediate departure was missing. No replay or policy update occurred.

Components cover 26,941/34,795 forecasts (77.43%), with no historical fallback.
Pooled ready MAE: raw 65.65 s, running-only 56.29, stops-only 66.09, split 50.98.
Split paired cell-mean MAE delta is -14.67 s [-16.11,-12.99]. Same-ready sign
errors fall 36.31% to 29.05% (negative), 28.90% to 23.36% (quota); paired changes
are -7.24 [-7.87,-6.61] and -5.53 [-6.20,-4.83] percentage points. All eight
cells improve; actual departure-pair sign errors also decrease in both cohorts.

The frozen advancement criterion passes. Preserve this split predictor without
window/threshold tuning; next use fresh demand days for passive independent
prediction/sign confirmation. Pending-stop coverage and factual holding-response
uncertainty remain. No allocation or closed-loop controller change is justified yet.
