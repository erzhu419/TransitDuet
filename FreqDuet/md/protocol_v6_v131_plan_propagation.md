# V131: read-only V130 delay and lower-response diagnosis

Read eight consumed V130 candidates and their V127/V123/V124 controls only.
No rollout, training, controller, budget or gate change. Scheduler CPU environment
on node001--006; return a small summary, keep per-trip/stop detail on the server.

Reconcile total launch delay into critical control wait, selected-vehicle supply
lateness and unexplained residual. Separate requested wait from wait that actually
blocks launch. For unchanged vehicle/prior-trip mappings, split supply changes
into previous launch, holding, other elapsed, visibility lag and reference clocks;
apply one common positive-part secant slope so clipping contributions sum exactly.
Keep remapped supply separate. Recover stop actions from cumulative holding,
match whole-trip holding and original V130 endpoints, then locate first changed
stop actions and major turnover propagation. Primary focus: the three split
cells with increased launch delay and actor differences on day2231004.

Six synthetic tests precede analysis. This is realized-path accounting, not
causal per-action value. Stop-arrival clocks do not substitute for missing
policy-query clocks. Decide the next experiment only after inspecting the result.

## Outcome (2026-10-19)

Task t195216 exited 0; six tests and all eight endpoint reconciliations passed;
no unexplained delay or remapping. Three split cells add 1.29--2.32 s/trip:
holding falls 0.55--1.22 s, but its clipped supply offset is only 0.013--0.057 s.
The favorable cell offsets +1.408 s control with -5.668 s supply change.
Its first command is trip18 +1 s at query3240; first changed stop action is
trip16/station5 0 -> 30 s, arrival3231 unchanged. Arrival is not policy-query time.

Next: passively replay the two day2231004 candidates, reuse baseline traces,
and save decision-time follower identity, prediction source, gaps and actions.
Require original endpoints/commands/timing exact, then distinguish normal action
thresholds from terminal/en-route identity switching. No tuning, training or
main switch. Final small summary (72280 bytes):
`freqduet/results_freqduet/protocol_v6_v131_plan_propagation/summary/status.json`.
