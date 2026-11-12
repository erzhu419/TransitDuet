# V413: Passive public-slot supply forecasts

Freeze V412's slow input, capped V406, fleet, timetable, randomness and gates.
Sample every public slot at 300 s before and immediately before the native
dispatch step at the public clock. Retain initial/unavailable, already-ready,
and earlier-queued slots separately. A sample does not require a holding query
or override. The 0 s sample is a nowcast, not advance warning.

Use the frozen V406 reference continuation without a distinguished current
command. Reuse clearance-block and historical-prior forecasts and both archived
demand-scenario definitions. Already issued commands are fixed countdowns;
only current observable state and completed history are used. No action search
or change to the live bridge. Save complete candidate IDs/clocks, prior slots,
model support and scenario labels on the server; join actual returns only
after the path ends.

Primary readout at 300 s: coverage, public-time supply misses/false alarms
against the observed-ready-only comparator, and first-return rank among the
complete compatible inbound cohort when no vehicle is ready and no earlier
slot is queued. Report rank errors at <=30 s and >30 s predicted separation,
also reporting their denominators. Use the first scenario as primary, the
second as sensitivity, and separate all results by lead and demand day. Do not
pool repeated snapshots as independent observations or choose a new threshold.
Observed return rank is not a counterfactual or a future action label.

Physical launch hold: this stage permits only zero-traffic checks. Proposed
next budget is one passive parity path on `2520208`, reusing the existing
capped reference; no fresh days until exact actions, physical clocks, CRN,
endpoints and old resource records match. Then eight reserved independent
slow days, one path each, subject to release. Failed paths consume slots.
Scheduler high/manual on CPU nodes only; sync only small summaries.

**Preflight.** `t214972` caught a skipped intermediate slot in the synthetic
clock fixture. After fixing the fixture to visit every tick, `t214974` passed
the four zero-traffic checks and found no reserved-day collision in the
case directories or 25 protocol manifests checked. No parity or confirmation
path has started. [Small preflight result](review/preflight02/results.json).
