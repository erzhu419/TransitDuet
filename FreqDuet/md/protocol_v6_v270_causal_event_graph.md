# V270: causal event graph and bounded rolling allocation

V269's allocation hypothesis is now implemented as a pure query callback,
not a V268 residual-policy search. The native controller and all historical
versions remain unchanged. This release permits synthetic server checks only.

## Inputs And Execution

`freqduet_causal_event_graph.py` accepts the existing current fleet capture,
published unlaunched roster, static directional routes, and same-query cursors:
next stop, remaining link fraction, executed H, legal actions, observed arrival
and APC boarding count. It does not read passenger destinations, latent queues,
future simulator durations, future vehicle assignments or evaluation labels.
The pure capture adapter does not install callbacks in the native simulator.

Run, alighting fraction and uncensored boarded/gap demand forecasts use only
visits whose completion AND label-availability clocks are strictly before query.
Three completed visits per directional stop are required; demand uses the latest
three and run/alighting means use the complete strict prefix. Missing/currently
unavailable fields or capacity-censored demand retain the original native request.
Current APC boarding is removed from the forecast queue; doors do not reopen
during holding. Forecast capacity excess stays waiting. Arrival order is rebuilt.

Executed H is recorded but never converted into slack or spending authority.
Current service and remaining issued holding are fixed. Public launches use the
observed physical FIFO pool, original roster and return+1 event ordering. There
is no terminal hold, new vehicle, reassignment command or speed authority.
The snapshot is before the query tick's drive: zero holding still consumes the
native DWELLING transition tick, and observed issued-command countdowns resume
from the previous tick. Simultaneous vehicle events retain physical fleet order.

The graph extends through active vehicles' next launches and completion of the
included future trips. The horizon is query plus the longest predicted route
upper bound (rounded run,104 nonterminal service,native transition ticks and up
to45 legal holding), plus360*fleet+1.
Include published releases through query+360*fleet. Incomplete modeled coverage
cannot produce a feasible allocation. Prediction boundaries stay fixed per query.

## Solver And Control

The solver uses the current node plus the earliest other active-trip controls,
up to6 variables, beam8,512 model evaluations and12000 events per evaluation.
Same-vehicle equal-total transfer proposals allow coupled moves without requiring
an individually feasible intermediate. These are proposals, not earned budgets.
No global optimality is claimed. Retain native unless a strict modeled passenger
gain preserves modeled CV2, first return, next launch and public launch delays.

`OnlinePlanner` reassembles at every query and warm-starts revisable future
proposals; only its current first legal command is returned. Unavailable support
falls back rather than reviving old proposals. The homogeneous control uses the
same graph, physical forecasts, constraints and solver budget; it substitutes
the query's mean active load for location-specific onboard exposure only.

## Sealed Full-Day Pilot (Not Started)

The machine-readable protocol is emitted by the synthetic runner. Arms are
unchanged exact360+bilateral025, graph/APC and graph/homogeneous-cost. Four
already-consumed development demand days2071001-4 cross two new runtime seeds
27011001-2:24 complete-day paths, plus2 engineering equivalence paths, at most8
concurrent. Failures count; no automatic retry. Seed12011 is the frozen legacy
execution anchor, not an actor-generalization experiment. Confirmation stays shut.

Physical release first requires native adapter/no-op equivalence and verified
policy-independent exogenous coupling. It is not authorized by synthetic success.
The joint gate uses actual once-only passenger time, full-day arrival CV2,
readiness and total public delay including control waits, unfinished service,
and actual holding/exposure. Paired crossed day/runtime intervals and both
future halves must support the frozen joint gate. An equal-budget homogeneous
control tests whether APC allocation adds anything beyond broad holding changes.
Failure closes this pilot; no forecast, weight or window repair on its outcomes.

## Limitations

This is a causal fluid forecast and bounded optimizer, not validated field ETA,
an exact simulator model or proof of traffic improvement. The support/censor
fallback may restrict use heavily. Native query-to-cursor/APC prefix wiring and
runtime cost require engineering preflight before any physical pilot release.
No full-day run, policy training, new frequency claim or main switch occurs here.
