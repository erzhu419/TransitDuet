# V412: Frozen Transfer Under Slower Running

Test supply pressure through the running-speed input, not fleet size. Freeze
V410 capped V406, its within-day transition reference (including counts, tilt
and seed), fixed10, demand at 1.2x, 12 buses, 262 trips, topology, timetable,
speed noise and evaluation horizon. In the slow arm only, subtract 0.5 m/s
from every hourly road-speed mean before the native speed draw and clipping.
This is a simulated input shift, not a promise that realized speeds are
uniformly 0.5 m/s lower. The controller sees its usual measured AVL and
completed history; it gets no future label or explicit treatment input.

Eight new demand seeds shared by the two settings, three policy arms per
setting, at most 48 complete-day paths. The two first-day tasks count as
formal cells. A physical-only gate verifies applied speed inputs, shared
generated population, 5,200 random-event pairs per setting, fleet and
native execution. It does not read outcome effects or change the endpoint.
All remaining jobs require the gate; failures consume starts with no retry.

Use the V410 holding (+/-3 s/trip) and bin-share (+/-2 percentage points)
gates and the same journey, actual CV2, readiness and public-delay outcome
tests separately in each setting. Preserve fixed10 contrasts. Report a
descriptive paired change in the capped-minus-transition effect across
settings on the common days; do not interpret it as a matched comparison
where either setting fails dose/composition. No pooling, tuning, training,
recalibration, main-policy switch, or cross-route/fleet-size claim.

Scheduler high/manual on node001--006. Only small summaries sync locally.
