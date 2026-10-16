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

## Outcome / Next Step

All shards/aggregate completed rc=0/retry=0: 160 groups, 384 exact V70 endpoints
and prefixes, zero uniform mismatches. Read-only audits `t165730`/`t165733`
passed 3/4 tests, adding no replay/update. Pulled 110,184 bytes of small JSON.
c123 0->45: +15min/day exposure deltas +194.75/-46948.09 person-seconds,
CIs [77.78,311.35]/[-69594.02,-27171.77]; 20/32 signs reverse at 15min,
12/32 at 30min. The five strong early-action contrasts realize 95.3-99.0%
of mean net benefit after reference return, chiefly through waiting.
For c123 0->45, later commands fall 34.375s on the target trip; actual day
holding changes -19.531s (CI spans zero). Target return/next launch are both
9s later; 62.06 actual launch times change versus 161.81 published plans.
No vehicles reassign. c125 30->0 instead returns 12.78s earlier, next launches
7s earlier and executes 34.06s less holding; all three conditional cost CIs
are negative. c51 0->5 exposure CI spans zero; c87 improves exposure despite
increased holding and a worse CV-squared point estimate. No blanket holding cut.
Common post-service tail deltas are zero for c15/c123/c125; c87's tail accounts
for 9.9% of net benefit. Full outputs: `summary`, `fleet_summary` (`6aa2f789b9`),
`fleet_tail_summary` (`1aed4f6c3d`).

Next: low-capacity full-day passenger/CV-squared/readiness action-increment
scorer, whole-demand-day exclusion, zero-delta controls, then reserved fresh
days/prefixes. No fitting launched here; actor/dual/main and V70 gates unchanged.

Limitations: posthoc consumed contexts/futures, all five from demand day
700006, and two original training seeds. Per-context future-bootstrap intervals
describe these conditional continuations; they are not independent-day or
selection-corrected control confirmation. Temporal decomposition does not
identify causal mediation. V70 gates and deployment stay unchanged.
