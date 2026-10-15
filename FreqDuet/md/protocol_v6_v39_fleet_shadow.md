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
