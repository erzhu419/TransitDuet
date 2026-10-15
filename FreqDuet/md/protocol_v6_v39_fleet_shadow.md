# V39: observable fleet-pool shadow forecast

Replace V38's nearest-departure margin with a pooled service forecast before
using it for control. Keep the original 12 vehicles, checkpoint/deployment
state, holding bins and causal upper replanning. This stage does not alter
policy actions or claim passenger improvement.

At both V37 decisions, freeze current AVL/APC, terminal inventory, issued
commands, route geometry, published unlaunched trips and known dispatch gaps.
Estimate segment movement, stop dwell/holding and complete-trip duration only
from already observed events. Forecast terminal returns, FIFO reuse, actual
launches and backlog exposure over 15/30/60 minutes. Do not clone simulator
state, use scenario randomness or inspect actual future vehicle returns.

Replay the 12 original actor references plus the dominant context's three
interventions, with exact V37 outcome matching. Calibrate all current-trip
returns, launch timing/window coverage, readiness and vehicle identity.
Separately report returns actually within 15 minutes; a prospective resource
model must place at least 90% within 45 s (the largest holding action) before
being considered action-resolution adequate. This is a development resolution
check, not an efficacy or safety guarantee.

Generate seven-by-seven resource-only shadow projections. The second decision
is selected from model-predicted service arrivals, including vehicles not yet
launched, never from the realized second identity. Historical continuation
holding is the second reference, not a future actor action. Passenger value,
future policy-path cascades and tail value are not present in this resource
model; no candidate is deployed based on its backlog score alone.

All tests, replay and aggregation use scheduler on node001-node006. Preserve
full traces on server and retrieve only small completion markers/summaries.

## Outcome and next step

Smoke `t153476` and batches `t153478-t153482` pass exact V37 matching; 15
formal replays use 4/2/3/3/3 workers across five nodes. Regression `t153483`
passes 23 tests; decomposition `t153485` passes nine focused tests, including
the additional holding-cut test and equal dominant-context pre-action snapshots.
Replay source is `10e3fa4ec7`; final analysis source is `4860229a6f`.

Second vehicle/trip identity matches in 12/12 reference contexts. Target-vehicle
return MAE improves from V38's 191.57 s to 126.44 s; all 282 current-trip
forecasts have MAE 121.67 s. For 112 returns actually within 15 min, MAE is
54.91 s, P90 absolute error 153.96 s, and only 63.4% are within 45 s: the
resolution check fails. Fifteen-minute dispatch MAE is 35.74 s for predicted
launches, with 95.5% launch-window classification accuracy; missing predictions
remain in the coverage report. No policy efficacy comparison was run.

Short-return holding/nonholding error MAEs are 50.01/61.55 s. Errors can cancel,
so neither a constant ETA correction nor holding-only adjustment resolves
forecast uncertainty. Retain the pool-matching interface as a research module;
next add causally calibrated return distributions and closed-loop continuation
holding/passenger dynamics before learning tail value or selecting a controller.
