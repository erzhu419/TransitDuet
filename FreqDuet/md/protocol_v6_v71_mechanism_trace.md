# V71: fixed V70 intervention timing and fleet mechanism

Freeze main, actor, upper/B200 deployment, dual and value. Reuse V70 prefixes
and all 32 continuations per context, without choosing favorable futures.
Fix native/candidate actions: c15 5/45; c123 0/20/45; c51 0/5/45;
c87 0/30; c125 30/0. Five contexts, 160 paired groups, 384 formal replays.
These include increased/decreased holding and joint-constraint failures.

Reuse the original system-credit integrator, adding only read-only timeline,
executed holding counters and physical launch observations. Every replay must
match its V70 cached endpoint and full original decision prefix. Keep V69
event uniforms and V70 future seeds unchanged. Do not fit or reselect actions.

Common cuts: before intervention, +15/+30 minutes, reference intervention-trip
physical return, reference vehicle's next physical launch when one exists,
and the fixed day end. Candidate return times are outcomes, not alternative
exposure windows. Preserve the service-cleared unserved-passenger tail.

Separate waiting/onboard exposure, actual holding, holding commands and other
commands after subtracting the one intervention. Track reference-trip return
shift, cumulative CV-squared, actual launch/readiness changes, physical bus
reassignment and published-plan changes. Plan changes alone are not launches.
Keep missing next launches explicit rather than replacing them with day end.

Smoke on c15/c123/c125 future 0: focused tests, cached V70 endpoint identity,
forced-native full trace identity and traced/uninstrumented full holding,
upper-policy and day identity. 13 smoke replays, 7 reused formally.
All tests, replays and aggregation run via scheduler on node001-node006.
Only small JSON/markers/logs are pulled; raw trace data remain on servers.

Execution: source `3b4cfe80b6`, server snapshot `source/v71_trace_20261016`.
Smoke `t165724` on node004 passed 21 tests and all 13 identity replays;
only its 53-byte completion marker was pulled. Formal shards `t165725`
(groups 0-80) and `t165726` (80-160) each use 80 workers, with node004/node005
preferred and node001-node006 allowed. `t165727` aggregates after both exports.
All 384 formal replay endpoints remain subject to the same V70 equality checks.

Limitations: posthoc consumed development contexts/futures, mainly one demand
day, and two original training seeds. Per-context future-bootstrap intervals
describe these conditional continuations; they are not independent-day or
selection-corrected control confirmation. Temporal decomposition does not
identify causal mediation. V70 gates and deployment stay unchanged.
