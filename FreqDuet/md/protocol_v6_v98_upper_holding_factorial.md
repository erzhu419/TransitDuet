# V98: frozen upper-by-holding factorial

V97 leaves passenger/readiness costs despite V96's genuine joint improvement.
Stop tuning the compensation split. Before lower system-credit redesign,
cross B200 versus exact 360 s fixed upper with preserve versus zero holding.
Keep all checkpoint weights, dual, deployment background and episode at B200.
The fixed arm installs the existing V6 exact schedule, not forced actual launches;
physical vehicle readiness still governs service. Native seven-bin masks remain.

Reuse V96 learned/preserve (960 paths) and V97 fixed/zero (32 shared day paths).
Add learned/zero once per actor/day (320) and fixed/preserve for all three event
samplers (960): 1,280 new full-day paths on the same ten actors and 32 consumed
days. The smoke must reproduce cached preserve and external fixed, exercise
preserve under fixed upper, and establish zero-law invariance to sampler repeat.
If any contract fails, fix plumbing before formal replay; do not reinterpret
different schedules as a successful factorial endpoint.

Report holding effects at each upper, upper effects at each lower and the exact
difference-in-differences. Repeat-average then crossed actor/day bootstrap;
shared deterministic zero paths are not independent replicates. Outcomes are
passenger exposure, journey, arrival CV-squared, readiness, holding, dispatch
gaps and completion. Retain positive and negative signs; no arm selection or
new admission gate. These interventions identify regime effects, not mediation.

If holding causes most passenger/readiness cost at both uppers, redesign lower
system cost/credit. If upper-without-holding already costs substantially more,
include upper dispatch objectives; an interaction requires coordination rather
than an isolated lower reward patch. Better journey at zero alone does not
justify promotion if regularity/completion are harmed. No training or main
switch in V98. Scheduler only on node001-node006; raw records stay on-server.

Smoke `t175833` on node004 passes the three initial tests and five full-day
paths under source `1a4c681b0a`: cached preserve, deterministic-zero repeat,
external fixed parity, fixed/preserve execution and shared physical contracts.
Zero bypasses the actor scorer; its inherited diagnostic counter initially
stayed zero, causing an unused risk-summary division warning. Source
`1eae342e9a` counts those decisions without changing the law; all four tests
pass in `t175850`. Formal tasks were still queued when redirected to this
source, and require both the full-path smoke and counter-test markers.

Live planning allocated 39 available physical workers: `t175844` [0,296) with
9, `t175845` [296,591) with 9, `t175846` [591,886) with 9 and `t175847`
[886,1280) with 12. Preferences are node002/004/005/006, with only node001-006
allowed and no hard pins. Aggregate `t175848` waits for all four shard markers.
No V98 outcome or policy qualification is claimed before aggregation.

Completed all tasks with exit 0 and no retries on the formal shards. Retrieved
only the 254,962 B summary. The same ten actors and 32 days are crossed, after
sampler averaging; deterministic zero corners are shared, not extra replicates.

| Difference | Journey min | Arrival CV-squared | Readiness s |
| --- | --- | --- | --- |
| Preserve minus zero, B200 upper | +1.98236 [1.76430,2.24277] | -0.12756 [-0.13633,-0.11885] | +410.52 [357.54,462.55] |
| Preserve minus zero, fixed upper | +1.98306 [1.76250,2.24611] | -0.12770 [-0.13644,-0.11913] | +410.35 [357.19,462.49] |
| B200 minus fixed, preserve | +0.00008 [-0.00814,0.00824] | +0.00007 [-0.00020,0.00042] | +0.15 [-1.15,1.59] |
| B200 minus fixed, zero | +0.00078 [-0.00126,0.00314] | -0.00008 [-0.00042,0.00029] | -0.02 [-0.07,0.02] |
| Holding-by-upper interaction | -0.00070 [-0.00884,0.00750] | +0.00015 [-0.00032,0.00065] | +0.17 [-1.12,1.61] |

Under both uppers, all ten actor means and all 32 day means have higher journey
and readiness costs but lower CV-squared with preserve. Every actor-specific
holding-effect CI has those same signs. Completed/unlaunched differences are
zero. An isolated seed's small CV interaction (12161) does not replace the
unresolved overall interaction or justify a new gate. Mean actual holding is
283.44/283.40 s per trip under B200/fixed; zero is exactly zero. At B200 upper,
extra onboard exposure is 1,717,345 person-s [1,655,070,1,777,520], extra waiting
677,005 [425,696,991,666]; these are outcome components, not mediation effects.

Decision: the fixed gap is almost entirely reproduced by the holding regime
contrast under these deployed uppers. Do not promote zero: it loses regularity.
Retain the V96 partial positive, stop split/p tuning, and prioritize lower
system-cost credit. Upper and interaction CIs crossing zero do not establish
equivalence or rule out benefits from a different upper. V99 first audits actual
upper authority and completed lower reward components with passive, endpoint-
matched replays; no new controller training or promotion.
