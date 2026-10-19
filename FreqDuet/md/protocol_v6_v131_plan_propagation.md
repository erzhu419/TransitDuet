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

Scheduler t195216 exited 0; six tests passed. All eight holding/delay endpoints
reconcile, with zero unexplained residual and zero vehicle/prior-trip remapping.
Split means, seconds/trip: direct control + selected supply change = total delay:

| Actor / day | Control | Supply | Total |
| --- | ---: | ---: | ---: |
| 12011 / 2231001 | +1.260 | +0.443 | +1.702 |
| 933013 / 2231001 | +1.092 | +0.198 | +1.290 |
| 12011 / 2231004 | +1.408 | -5.668 | -4.260 |
| 933013 / 2231004 | +1.878 | +0.447 | +2.324 |

In the three delay-increasing cells, whole-trip holding falls 0.55--1.22 s,
but its clipped supply contribution is only -0.013 to -0.057 s. The favorable
cell relies on subsequent holding and turnover feedback, not cheaper commands.
Its first command is trip18 +1 s at query3240; the earliest changed stop action
is trip16/station5 0 -> 30 s, with arrival3231 unchanged. Arrival is not query
time: do not infer a before-intervention action or a causal one-action effect.

Next: passively replay only the two day2231004 candidate paths, reuse existing
baseline traces, and save decision-time follower identity, prediction source,
gaps and actions. Require original endpoints, commands and timing arrays exact
before interpreting threshold versus terminal/en-route identity switching.
No budget/gain tuning, training expansion or main switch. Summary: 72280 bytes,
`freqduet/results_freqduet/protocol_v6_v131_plan_propagation/summary/status.json`.
