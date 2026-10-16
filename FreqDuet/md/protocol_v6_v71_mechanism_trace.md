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

Both shards and aggregate completed rc=0/retry=0: 160 groups, 384 cached
endpoints and prefixes exact, zero event-uniform mismatches. Read-only audits
`t165730`/`t165733` passed 3/4 tests; no added replays or model updates.
Only 110,184 bytes of markers/summary JSON were pulled in this collection.
Mean cumulative passenger exposure deltas (person-seconds; negative is better):

| Context / action seconds | +15 min | +30 min | Day end |
| --- | ---: | ---: | ---: |
| c15 5->45 | -30.41 | -698.78 | -34685.50 |
| c123 0->20 | -23.97 | -181.28 | -28909.09 |
| c123 0->45 | +194.75 | -83.28 | -46948.09 |
| c51 0->5 | -26.47 | -155.97 | -1235.19 |
| c51 0->45 | +48.72 | -876.88 | -36144.19 |
| c87 0->30 | -40.69 | -808.72 | -26763.16 |
| c125 30->0 | -593.28 | -1721.81 | -2427.38 |

For the five strong early-action contrasts, 95.3-99.0% of mean net improvement
occurs after reference return. c123 0->45 has 15-minute/day CIs [77.78,311.35]
/ [-69594.02,-27171.77]; 20/32 future signs reverse at 15 minutes, 12/32 at 30.
Its later commands fall 34.375s on the target trip and 64.531s system-wide;
actual full-day holding changes -19.531s (CI crosses zero), while the target
return and next launch are both 9s later. Actual launch times change on 62.06
trips on average, not the 161.81 published-plan changes. No vehicles reassign.
c125 instead returns 12.78s earlier, next launches 7s earlier, and executes
34.06s less holding; passenger/CV-squared/readiness conditional CIs are negative.
c51 0->5 passenger CI crosses zero; c87 improves exposure despite +23.44s
actual holding and a worse CV-squared point estimate. No blanket holding cut.

Common post-service waiting-tail deltas are zero for c15/c123/c125; c87 has
-2645.44 person-seconds there (9.9% of net benefit), c51 0->45 has +381.59.
The dominant early-action benefit is not explained by the cleared-service tail
or an earlier target-vehicle return. Full outputs remain in `summary`,
`fleet_summary` (source `6aa2f789b9`) and `fleet_tail_summary` (`1aed4f6c3d`).

Next fit a low-capacity full-day action-increment scorer on consumed development
labels, predicting passenger/CV-squared/readiness differences directly rather
than absolute return or short-window cost. Exclude whole demand days in offline
validation, retain zero-delta controls, and reserve fresh days/prefixes for later
confirmation. No fitting launched here; actor/dual/main and V70 gates unchanged.

Limitations: posthoc consumed contexts/futures, all five from demand day
700006, and two original training seeds. Per-context future-bootstrap intervals
describe these conditional continuations; they are not independent-day or
selection-corrected control confirmation. Temporal decomposition does not
identify causal mediation. V70 gates and deployment stay unchanged.
