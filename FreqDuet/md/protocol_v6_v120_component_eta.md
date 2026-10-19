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
