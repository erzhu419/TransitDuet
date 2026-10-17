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
