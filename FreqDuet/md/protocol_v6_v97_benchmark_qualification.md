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
