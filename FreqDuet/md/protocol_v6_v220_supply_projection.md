# V220: supply-constrained headway projection

V219's native preview is exact but admits no +30 proposal: whole curves advance
unfunded departures. Keep V219 sealed. V220 changes the projection geometry,
not the raw coefficient, fleet, cadence, actor or lower action permissions.

At each archived native fresh query, reproduce native pending clocks and
headways from the frozen runtime, direction-local EMA and query-time state.
Preview the SAME clipped native+30 raw request with EMA.35. Project its desired
HEADWAYS by unweighted squared distance, using scipy SLSQP with analytic
gradient, into the following joint feasible set:

- Native anchor, other direction, outside-horizon and unsupported cumulative
  departure clocks stay exactly native. Only currently ready-backed non-anchor
  slots in the public900s cycle may advance, never before the query clock.
- Reserve compatible ready vehicles once in pending chronological order,
  including earlier due trips and anchors, using the frozen V219 resource law.
- Keep native absolute and delta headway bounds. Keep each native rolling900s
  block's closing departure exactly native, hence its zero-sum headway budget.
- No continuous or executable clock may become later anywhere. Recompute
  adjacent target headways from jointly solved clocks; do not mask times after
  projecting or leave stale targets. No request means the native schedule.

Represent only permitted cumulative shifts as variables; fixed shifts are
identically zero, including block endpoints. The native schedule is feasible.
This expands the reachable curve shape compared with the frozen scalar basis;
it is an explicit method change, not unchanged-native projection equivalence.

Current budget: only the four consumed V216 captures, all fresh native queries
including warmup for reference equivalence, and the single +30 preview after
7200s. Use past observed launches only; no future return labels, outcome-based
selection, gain/weight sweeps, new replays, training or futures. Retain every
context and rejection. Tests check constrained recovery from V219 rejection,
resource reservation, closures, headway consistency, nonmutation and identity.
Every context must contain a ready-funded executable advance before any later
four-arm replay budget can be considered; this job does not authorize it.

V107 remains the strong control reference. Lower task law stays V219; all
predictors, calibrations, filters, actors, critics, duals and main stay frozen.
Run tests/static archive analysis via scheduler high/manual on node001--006;
return only a small JSON summary. Static support does not establish service
benefit or a frequency-learning contribution.

## Static result and next decision

High/manual t211219 on node006 exited0; source3e2f53b0d4 passed18 tests.
All352 native snapshots reproduce exactly, including executable ticks. Four
contexts admit3/8/3/8 queries versus0 for the unconstrained native+30 curve.
Every admitted advance is1 executable second and ready-backed. Fixed clocks
are exact, headway error0, maximum rolling-budget residual7.28e-12s, and no
continuous/tick delay or outside-cycle advance occurs. Only4155 bytes returned.
Static admission passes, not efficacy. Next preregister a bounded four-arm
execution smoke with an upper-authority gate FIRST: verify that these small
plan advances cause actual earlier launch on the same reserved vehicle before
spending the lower/joint budget. Keep all contexts and the same +30 request;
do not increase gain to manufacture a larger signal. Main/futures stay frozen.
