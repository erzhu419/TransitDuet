# V36 joint-endpoint space and V37 two-decision pilot

The existing V36 bank is checked first: journey, headway CV and fleet readiness
delay must each be no worse than the same-context actor reference. At least one
strict recorded improvement defines a Pareto-improving alternative. Report
effect sizes as well as incidence; tiny CSV-level changes are not service gates.
The hindsight journey minimum under these joint constraints is not a deployable
policy, an independent confirmation, or evidence of superiority to fixed.

Before reading this audit, the V37 pilot contexts are fixed to current main,
training seeds 12011/933013, scenario seeds 45007/934011, target times 2/6/10 h,
lower actor 200, upper actor 200, and unchanged B200 deployment background:
12 contexts, selected by identifiers and time rather than outcomes.

The first decision is the original V36 eligible decision. The second is the
next eligible decision at the same stop and direction on a different vehicle.
It is selected causally after the first intervention, so its identity may vary
with the first action. Enumerate the existing seven bins at both decisions.
For each fixed first action, compare exact second-decision prefixes across all
seven second actions. A second-action noop must reproduce its first-only V36
branch; the full actor noop must reproduce the original reference. Keep upper
actor/configuration fixed; subsequent replanning remains a causal policy response,
not an externally frozen timetable. No action authority or fleet resources change.

Run seven first-action shards per context, nine full rollouts per shard
(reference, first-only, seven pairs): 84 shards, 756 rollouts, plus a separate
small smoke. Compare the 49-pair hindsight feasible journey minimum with the
best single intervention at either decision, not just the first-action bank.
All contexts remain in the report. Small-pilot results describe finite-bank
improvement space; they do not validate a learned controller or a repaired paper.

All tests, audit, smoke, rollouts and aggregation use scheduler on node001-node006.
Only logs and small summaries return; raw records and checkpoints stay on server.

## Execution

V36 audit `t153197` completed after four regression tests (`t153196`): all
864 contexts retained. Main has Pareto alternatives in 66/144 early and 72/144
late contexts; all three endpoints improve strictly in 33/144 and 59/144.
Late hindsight selection gives journey -0.022425 min, CV -0.001209 and readiness
-2.559160 s on average. Median journey change is -0.0000185 min: gains are
heterogeneous, and these selected effects are not controller-performance estimates.
Four summary files totaling 38,862 bytes were retrieved.

V37 source `2a02dff5aa` passed 16 tests and a five-rollout smoke (`t153199`),
including V35 reference, V36 first-only, exact first/second prefixes and second
actor noop. Formal CPU batches `t153201-t153205` use scheduler-sized workers
19/8/19/19/19 on node001/node002/node004/node005/node006, one parallel wave.
Aggregation `t153206` waits for five small `batch_done.json` exports; per-shard
records and ready markers remain server-side. All five batches and aggregation
completed with exit code zero: 12 contexts, 588 pairs and 756 rollouts; all
prefix, V36 first-only and actor-noop checks passed. The five exports totaled
3,486 bytes; three result summaries totaled 17,433 bytes. No checkpoints or
raw rollout tables returned locally.

## Outcome And Next Step

On the same 12 contexts, hindsight feasible journey minima give:

| Comparison | Journey delta, min | CV delta | Readiness delta, s |
| --- | ---: | ---: | ---: |
| Best single at either decision vs actor | -0.023546 | -0.001825 | -2.777354 |
| Joint vs actor | -0.042266 | -0.003233 | -5.557888 |
| Joint constrained not to harm best single | -0.018622 | -0.001425 | -2.713740 |

Joint Pareto improvements exist in 9/12 contexts versus the actor and 5/12
versus the selected best single; strict three-endpoint improvements exist in
7/12 and 3/12 respectively. About 90% of the incremental journey gain comes
from one context (train 933013, scenario 45007, 2 h); two other improvements
are below 0.0004 min. These are localized hindsight opportunities, not broad
controller gains or confirmation against fixed. Next explain that operational
context and its joint actions, then test a causal joint-action score using
deployment-visible state before launching a training sweep. Critic calibration
and the cross-trip objective remain unresolved.
