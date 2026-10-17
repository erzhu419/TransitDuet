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
