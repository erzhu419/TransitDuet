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

The preflight stage permitted only zero-traffic checks. The parity stage has
one passive path on `2520208`, reusing the existing capped reference. No fresh
days until exact actions, physical clocks, CRN,
endpoints and old resource records match. Then eight reserved independent
slow days, one path each, subject to release. Failed paths consume slots.
Scheduler high/manual on CPU nodes only; sync only small summaries.

**Preflight.** `t214972` caught a skipped intermediate slot in the synthetic
clock fixture. After fixing the fixture to visit every tick, `t214974` passed
the four zero-traffic checks and found no reserved-day collision in the
case directories or 25 protocol manifests checked.
[Small preflight result](review/preflight02/results.json).

**Parity launch.** `t214975` used the single path allowance on `node004`,
scheduler high and manual, with no automatic retry. The existing V412 archive
is the reference. [Parity result](review/parity/results.json).

**Parity acceptance.** `t214975` passed exact actions, timing, physical
execution and endpoints. Read-only verification `t214981` matched all 12
archived streams and all 5,200 random events, ignoring only wall-clock
`graph_queries.total_s`. The 300 s and 0 s slots matched the public roster
exactly: 260 and 262 snapshots, with complete inbound forecasts on 242 and
243 respectively. Two advance snapshots precede simulation time zero;
unsupported in-range snapshots remain in the record.
[Verification result](review/verify/results.json).

**Confirmation release.** `t214982`--`t214989` are the eight reserved days
`2530001`--`2530008`, one unchanged slow capped path per day, scheduler high
and manual. All eight wrote a physical-start receipt on nodes 002, 003, 004,
and 006. Only the passive sidecar is new. No automatic retry, no policy change,
and no release beyond these eight paths.
[Submission receipt](submission_confirmation.json).
