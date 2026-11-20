# V505: What does stop holding add?

Freeze the four V504 aligned upper actors, and reuse its 16 paired evaluation
days and the archived joint actor results. Replan every 900 s from the state
induced by each arm; never replay the other arm's timetable. Compare
learned holding to (i) zero discretionary holding and (ii) the existing
causal one-step Manual holding scorer with the same six bins and budget.
The roster, fleet, estimators, and projection do not change. No training.
This is mechanism attribution on the V504 evaluation cohort, not a second
independent confirmation. The two paired contrasts are reported separately
with the V504 crossed bootstrap and unchanged joint criterion.

For the already identified 50314/53020003 tail, replay aligned and NoFreq
once with passive stock and dispatch instrumentation. Require original day
endpoints and scenario tape to match exactly, not wall time. Sample physical
available vehicle counts at 60 s (and first denied attempts at exact event
time); collect first eligible, first denial, actual release, physical bus ID
and the previous trip and return time. Outcomes do not enter any controller.
Count queries with any bins removed and queries whose unconstrained
preference at the same state was masked. The second count is a static
action diagnostic, not a no-budget trajectory comparison.

As an additional comparator, use the V504 Manual holding scorer but choose
the same 15 upper actions by deterministic rolling forecast. Project each
candidate with the native timetable projection. Forecast the next 8 trips
per direction, allocating physical buses by predicted ready times and
feeding those vehicles to the opposite terminal after one predicted trip.
Use causal position, residual dwell/hold, recent directional nonholding
times and last completed minute APC counts. Score expected departure-gap
waiting and predicted vehicle-readiness delay in passenger-seconds; the
delay coefficient is one nominal headway's arriving passengers. Use a
120 s simulated hard interdeparture floor and retain irrevocable trips.
This is a matched model-based comparator inspired by joint rescheduling
work, not a reproduction of the published nonlinear program.

Start four server smoke paths: the two passive anomaly replays and one
old-day zero-hold/rolling-rule path each. After exact trace and engineering
checks, start 208 formal paths (128 intervention paths, 16 rolling
paths and 64 passive aligned replays for budget counts). Aligned replays
must reproduce all archived endpoints exactly on each paired cell.
At most 26 concurrent scheduler high tasks, no automatic retry;
no new seeds, demand days or controller updates. Only compact summaries
and the two selected case traces come back to the workstation.

Engineering amendment, before any formal launch: t224998 and t224999
completed their simulated days but failed while serializing NumPy integer
trace fields. Count those launches. With native output types, permit one
manual replacement of each exact passive path, for a total smoke ceiling
of six. Keep the completed zero and rolling smoke paths. No other changes.

Independent field audit (does not calibrate or select the controllers):
use the agency's October 2025 weekday Route 111 stop events and modal
start/end pair to check directional operating times. Assess approximate
same-vehicle terminal turnaround separately using the first full October
workweek (Oct 6--10, 2025) in Cornell Tech's public Bus Observatory GTFS-RT
archive. Require both observed terminal stop events for both trips, the
same public vehicle identifier, consecutive observed trip IDs, opposite
directions, and a positive gap of at most 60 minutes. Use first observed
arrival-stop occupancy and last observed departure-stop occupancy.
Minute snapshots are not driver or bus-ready records; unobserved terminal
events, cross-route assignments, and longer gaps are not imputed.

Field-only follow-up: the destination stop updates were missing and the
two trip-ID namespaces differed. Link the same day's trips in the same
direction using the origin and at least one common interior timepoint,
each within 90 seconds, keeping reciprocal unique matches only. For
consecutive opposite-direction trips on a vehicle, use the agency endpoint
and next origin times; retain positive gaps of at most one hour. These
time-matched gaps are separate from the direct snapshot test.
