# Stage156: Service Allocation Authority

Stage155 propagated delayed credit but did not produce a consistent own-control
cost gain. Replace the scalar departure phase interface before further training.

Generic `budgeted_time_points` allocates bounded service intervals while preserving
the number of events and both window endpoints. Native execution commits six
departures per direction once, changes intervals within 240-480 s, and passes
each executable interval to the frozen lower. This changes service allocation,
not the trip budget; cumulative internal shifts need not stay within +/-120 s.

Two registered Stage155 one-step checkpoints (roots347/359), no outcome-based
selection. Six conditions on twenty paired scenes/root: reproduce source learned
dispatch, reproduce source zero-upper with nominal plan, frontload, backload,
causal harmonic forecast, reversed forecast adjustment. The forecast preference
is inverse square-root arrival rate, normalized to the original window duration.
Only the fitted current causal state is queried, never future realized demand.
All source/neutral scenes must reproduce before alternative plans execute.

240 frozen episodes, 14,731,200 native ticks, zero training updates. Scheduler
node001-006, code-only staging, server checkpoint reuse, compact JSON outputs.
Judge physical restricted cost/wait, fleet, completion and unserved separately;
record actual-versus-planned intervals and fleet-cap release delays.
78 focused tests pass: real dispatcher execution, unchanged nominal endpoints,
bounded/communicated lower goals, fixed commitment, tail windows, paired analysis,
code-only scheduler placement and preserved one-step optimizer behavior.

Run `native_transit_service_allocation_stage156_frozen_20261010_r1`:
t141325/root347 RUNNING on node006; t141326/root359 RUNNING on node005.
Server reproduction and alternative-plan performance are pending.

## Limitations

This qualifies an upper action representation, not learned upper superiority.
The lower was trained at 360 s, so new goals are distribution-shifted. Its reward
depends on the supplied goal and is diagnostic, not a common-objective reward
improvement claim. Anchored windows cannot change total daily service frequency.
No learning expansion before physical execution and useful control authority are
examined; a forecast win alone would not establish learned planning value.
