# V101: native action differences

V100's small aggregate bias does not establish correct action ranking. Do not
distill its critic into the actor or retune holding penalties on that evidence.

Freeze B200 actor/critic/dual/upper and replay the same ten actors, two consumed
days and r0. Capture the first observable two-sided, nonterminal seven-bin
decision after 7200 and 21600 s: 40 prefixes from 20 native references. Verify
reference endpoints/actions against V100. For each prefix use four independent
post-prefix innovations, all seven actions once, then native feedback: 1120
arms. Keep pending arrivals and day latents; use existing event-keyed CRN.
Require exact causal-state/lower/upper prefixes before each intervention.

Compare Q action differences with remaining-trip native soft reward and cost
differences, not absolute levels. Evaluate the fixed critic's Boltzmann-minus-
actor distribution using all seven measured arms, including current entropy.
Keep its existing pessimism charge separate from physical passenger cost.
Record full-day passenger exposure, CV, readiness and holding alongside local
return. Average futures/times before crossed actor/day inference. Split-future
curves describe replication; they select no action or weight.

Next decision: realized local gain with system harm supports objective repair;
failed local gain supports critic action-difference repair. Both can coexist.
No policy promotion. Smoke must reproduce the forced-native full timing path
and both extreme forked actions before formal release. All tests/replays/stats
use scheduler on node001-node006; only compact summaries return locally.

Limitations: consumed days and four conditional futures; no fresh-day efficacy
or full-day policy claim. Single-action effects cannot be summed into a new
controller's daily gain. No search over gates, mixtures or reward weights.

Source `35c958b25e`. Prepare `t176178` passes three tests and reviews V100;
retrieved only its 3,061-byte summary. Native-reference shards `t176180-t176184`
finish all 20 cached-native checks; their five markers total 135 bytes. Initial
idle-capacity action shards `t176185-t176189` and aggregate `t176190` were
cancelled while still queued after other projects occupied CPU capacity.
Replacement action shards `t179080-t179084` cover [0,143), [143,448), [448,692),
[692,916), [916,1120), with 7/15/12/11/10 workers across node001/002/004/005/006.
Source, 1120-arm budget and all diagnostic contracts are unchanged. Smoke and
reference paths are reused, not repeated as additional formal evidence.
Aggregate `t179085` waits for all five replacement shard markers.
No V101 effect is claimed before completion.
Smoke passed on node004: native cache identity, forced-native full timing,
exact prefixes, seven-bin feasibility, completed-trip returns, frozen critics
and both extreme forked actions with common event uniforms. Only its 69-byte
marker is retrieved; this admits the matrix, not a controller or effect claim.

Completed: 1120 arms and aggregate `t179085` exit 0, formal retry 0. Retrieved
only the 215247-byte summary, then the 136456-byte cached attribution from
`t179113` (source `4646cd3334`, three tests pass, no new rollout/update).
Native soft gain +0.002448 [-0.005136,+0.011870]; fixed-charge gain +0.004722
[-0.002715,+0.015691], versus predicted +0.004407. Passenger, CV and readiness
intervals all cross zero. Reward/cost action-difference MSE exceeds observed
label-mean variance; reward predicted/observed rank correlation 0.550 versus
split-future 0.861. Thus mean calibration does not establish action accuracy.
Retain V99's separate objective-conflict evidence; do not claim V101 proves
aggregate system harm. V102 tests a fixed-budget, cross-day critic-only repair
without deploying its forecasts or changing actor/dual/main.
