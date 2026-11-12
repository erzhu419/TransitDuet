# V412: Journey and Regularity Transfer Under Slower Running

All 48 traffic paths completed without retries. The same eight new demand
days were used at normal and shifted speed; the generated populations agree
across settings. Each day/setting passed the fleet, original endpoint,
no-early-departure, 5,200-event pairing, and applied speed-profile checks.
The initial preflight t214939 failed on the input directory with zero traffic;
t214940 passed after using the native runtime's data root. The first six
traffic paths passed the physical release gate and are included in the 48.

Capped V406 minus the frozen within-day transition rule, with paired 95%
day-bootstrap intervals:

| Endpoint | Normal speed | Speed input -0.5 m/s |
| --- | ---: | ---: |
| Holding, s/trip | -0.131 [-0.728, +0.549] | +1.281 [+0.157, +2.593] |
| Journey, s/person | -3.165 [-4.637, -1.791] | -7.613 [-13.915, -1.194] |
| Actual CV2 | -0.006226 [-0.009800, -0.002160] | -0.041318 [-0.052375, -0.030613] |
| Readiness, s/trip | -1.844 [-3.177, -0.345] | -4.010 [-11.047, +3.550] |
| Public delay, s/day | -483.0 [-832.3, -90.5] | -1050.5 [-2894.3, +930.1] |

Both settings match mean holding and all three bin shares within their
frozen margins. Normal passes all four outcomes; slow passes journey and
actual CV2 but fails both delay nonharm gates. Readiness improves on 5/8
slow days; on day 2520208 it worsens by 16.725 s/trip while CV2 improves
by 0.0390. Slow journey improves on 6/8 days, with mean waiting down
7.461 s/person and onboard time changing -0.152 s/person (interval crosses
zero). The P95 journey interval also crosses zero; do not infer a tail benefit.

The shared-day difference in the capped-minus-random CV2 contrast is
-0.035093 [-0.048822, -0.023262] (slow minus normal). The journey and
readiness interactions cross zero. The CV2 contrast widens primarily
because the randomized comparator deteriorates against fixed10 under slow
running (+0.039102, 8/8 days worse). Capped versus fixed10 at slow speed
reduces journey 8.748 s/person and readiness delay 17.487 s/trip on all
eight days, but uses 3.893 s/trip less holding and its CV2 interval crosses
zero. This does not establish a CV2 benefit over the strong fixed10 rule.

**Decision.** Preserve the matched-dose journey and regularity evidence
under slower running; do not claim a transferred joint circulation benefit
or switch the main policy. The useful follow-up is to read existing paired
return and launch records on all eight slow days, isolating where readiness
goes in the opposite direction to regularity. No replay or change to the
original gates is needed for that audit. One corridor, fixed fleet,
eight common days, and a shift to the simulated input speed means bound the
claim. The controller and comparison were not recalibrated.

[Frozen protocol](PROTOCOL.md) | [Small aggregate](review/aggregate/results.json) |
[Paired interaction](review/aggregate/interaction.json)
