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
