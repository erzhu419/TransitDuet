# V97: frozen-candidate benchmark qualification

V96 replicated both joint contrasts. Freeze preserve and p; stop tuning the
compensation split. The next question is residual lower-actor drift and strong
fixed-policy tradeoff, not another same-gate confirmation.

Reuse all 960 V96 actor/day/sampler cells. Run one lower-40 categorical actor
on each cell, under unchanged B200 upper actor, critics, dual, deployment and
episode stage. Reuse the existing early law and event sampler; this is not the
complete B40 checkpoint or a newly trained actor. Run the existing exact
360-second/zero-holding fixed baseline once on each of the same 32 days, using
the verified optimized V6 runtime and the same immutable main config. Its upper
policy deliberately differs; physical fleet, demand tape and horizon must match.
There are 992 additional paths, not 960 independently trained fixed policies.

For fixed, collect the same physical per-step waiting/onboard/backlog rates
with SystemCreditStream and its fixed-horizon tail. Do not substitute composite
or reconstruct exposure from rounded mean journey. Use existing external
schedule/fleet logic, not the legacy elastic-fleet checkpoint-selection script.
Match the runner's six-decimal journey/readiness/holding reporting convention.

Smoke on a consumed V96 day: new lower-40 recording must reproduce its existing
early-law event/holding path; native late must reproduce cached V96 metrics;
fixed recording must leave every original external result unchanged. Match
existing scenario tape ID, generated demand, protocol clocks and physical fleet
across paths. No new hashes or provenance framework. Smoke gates formal cells.

Average samplers per actor/day and retain the 5,000-draw crossed bootstrap.
The one fixed result per day is shared across actors; shared day resampling
must retain that dependence. Report late-minus-lower40, preserve-minus-lower40,
preserve-minus-fixed and control contrasts, actual holding, passenger exposure,
CV-squared, readiness and completion. Check the additive identity between drift,
relief and residual; do not use unstable recovery ratios or select a winner.

A benefit against B200 does not establish full drift repair; unresolved or
harmful differences to early/fixed remain explicit. If sizeable residual costs
remain, return to lower system-credit/objective design, not another p/split
adjustment. If comparisons pass, retain qualification without changing main.
All actors/critics/dual remain untrained; no controller admission is authorized.
This consumed-day benchmark is not a new independent V96 confirmation.

All tests/replays/statistics use scheduler on node001-node006 in the dedicated
CPU env. Only smoke/shard/summary artifacts sync locally; all checkpoints,
per-cell results and timing arrays remain on-server.

Frozen source `e5da22c51b`; smoke `t173135` passes four tests and five full-day
paths on node004. Original early sampling/holding paths, cached V96 native
metrics, passive fixed recording and the same-day physical contract all match.
Retrieved only its 126-byte marker. Four live-planned 83-worker shards
`t173136-t173139` cover [0,248), [248,496), [496,744), [744,992), soft-preferred
to node002/004/005/006 and allowed only on node001-node006. Aggregate `t173141`
waits for all four small completion markers. No baseline result is claimed yet.

All six tasks completed with exit 0 and no retries. Retrieved only the 213,844 B
summary. Samplers are averaged within actor/day before the crossed bootstrap.

| Difference (candidate minus reference) | Journey min | Arrival CV-squared | Readiness s | Holding s/trip |
| --- | --- | --- | --- | --- |
| B200 minus lower40 | +0.513 [0.153,0.899] | -0.00548 [-0.00915,-0.00216] | +218.59 [83.92,349.73] | +34.92 [12.54,56.98] |
| Preserve minus lower40 | +0.244 [-0.051,0.547] | -0.00814 [-0.01197,-0.00477] | +116.13 [8.22,219.89] | +21.34 [1.90,40.50] |
| Preserve minus fixed | +1.983 [1.764,2.245] | -0.12763 [-0.13642,-0.11903] | +410.50 [357.51,462.54] | +283.44 [275.98,290.40] |

The same-day additive relief is 0.269 min journey, 102.46 s readiness and
13.58 s/trip holding (V96); sizeable residual costs remain. Journey's residual
crossed CI includes zero: this establishes neither harm nor equivalence. Of ten
actor means, preserve is worse than lower40 on journey for seven and readiness
for eight; of 32 day means, 31 and 32 respectively are worse. Seed 12143 already
improves journey under late versus early, so a per-seed recovery fraction is
not a uniform scientific estimand. Preserve versus fixed has worse journey and
readiness but better CV-squared for every actor and every day. All completion
differences are zero. Fixed has 32 shared day paths, not 960 independent runs.

Decision: retain V96's positive frozen candidate, but reject full lower-drift
repair or strong-fixed superiority. Do not tune p/split again. Before changing
system credit, V98 crosses exact fixed versus B200 upper with preserve versus
zero holding, reusing the two existing corners. This isolates controllable
policy-regime effects and their interaction, not natural mediation. Main stays
unchanged; these consumed-day diagnostics are not a fresh confirmation.
