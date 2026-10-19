# V119: completed stop clocks and same-object action signs

Read only the eight exact V118 funded paths. No replay, new days, fitting,
training, quota or main-policy change. Tests and analysis run via scheduler on
node001--node006; only the final small JSON returns.

For journey-speed ETA, elapsed time is measured from actual physical launch.
Index holding, ready-minus-arrival service and execution ticks at departures
strictly before the decision. With progress = speed * elapsed, each completed
clock contributes spatial_gap / progress * clock to the original ETA. Evaluate
raw, minus completed holding, minus service, and minus both as analytical
sensitivities, not proposed deployment predictors. Preserve other ETA sources.

Report ETA/ready errors and fixed-path restoration-sign disagreement, corrected
false positives and lost true benefits, for all forecasts and the two frozen cut
cohorts. Compare paired actor/day means using the existing crossed bootstrap.
On followers labelled in transit, decompose ETA error into extrapolated minus
actual future holding/service/ticks and remaining timing error. Future events
are labels only; pending-stop and exact-event-boundary cases retain coverage
but have no fabricated future-component labels.

Decide whether completed stop clocks explain wrong action signs before designing
any allocation candidate. Better MAE alone does not qualify a control change;
historical stop time can predict genuine future stops. Local signs do not prove
closed-loop system benefit, and these consumed days are diagnostic data.

## Outcome and next decision

Source `5e66c3b14f`, scheduler `t189900`, node006: five tests passed, eight
paths read, exit 0. Only 150,068 bytes returned. No replay or policy changes.
Pooled ETA MAE is 65.258 s raw, 64.340 minus holding, 69.361 minus service,
80.155 minus both. The minus-holding paired cell-mean change is -0.918 s,
95% crossed interval [-1.229,-0.584].

Minus holding changes same-ready sign disagreement from 36.31% to 36.65%
(negative cuts) and 28.90% to 32.43% (quota extras). It corrects false positives
on 5.98%/9.26% of these decisions but loses true benefits on 6.32%/12.79%.
For 25,021 in-transit labels, holding extrapolation 21.92 s accompanies real
future holding 19.22 s; service extrapolation 56.65 s accompanies real future
service 40.60 s. These timing means do not establish causal error shares.

Reject direct clock subtraction for allocation; quota/main remain frozen.
Next model remaining running, service and holding separately from completed
prefixes, with same-object ready/action-sign validation. Do not tune subtraction
strength on these consumed days or launch a changed-control matrix from MAE alone.
