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
t141325/root347 and t141326/root359 are DONE. All 240 episodes qualify;
40 original learned and 40 nominal/zero-upper episodes reproduce exactly.

Roots347/359 versus nominal allocation:

| Plan | Restricted cost delta | Restricted wait delta (min) |
| --- | --- | --- |
| frontload | +0.180186 / +0.107543 | +0.18190 / +0.43135 |
| backload | +0.300927 / +0.240062 | +0.66970 / +0.53245 |
| causal forecast | -0.000767 / -0.002681 | -0.00720 / -0.02870 |
| reversed forecast | +0.022797 / -0.018933 | +0.01480 / +0.01880 |

Plans are executable, not merely requested: fixed shapes change mean absolute
headway by 60.15 s; causal forecast changes it by 2.105/2.024 s. Its average
actual-versus-target error is 0.022/0 s. Forecast-minus-reversed waiting improves
both roots (-0.0220/-0.0475 min), but total cost remains fleet-step sensitive.
Forecast gains vary by regime and are small, not confirmation evidence.

Decision: proceed to a bounded learned residual-plan test with frozen lower,
explicit causal plan context, and the SAME final physical service-cost objective.
Use undiscounted finite-episode prefix-cost differences rather than the old
linear fleet-exposure surrogate. Compare own learned/zero/constant residuals
against pure causal forecast; no full joint-HRL or promotion claim yet.

## Limitations

This qualifies an upper action representation, not learned upper superiority.
The lower was trained at 360 s, so new goals are distribution-shifted. Its reward
depends on the supplied goal and is diagnostic, not a common-objective reward
improvement claim. Anchored windows cannot change total daily service frequency.
No learning expansion before physical execution and useful control authority are
examined; a forecast win alone would not establish learned planning value.
