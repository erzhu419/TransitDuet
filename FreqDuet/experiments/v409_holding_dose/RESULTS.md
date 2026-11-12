# V409: Same-Dose Benefit with the 10-Second Menu

All 82 charged traffic paths completed without retries (two old-day calibration
paths, 80 confirmation paths on 16 new heavy-demand days). All five arms on each
day share 5,200 random events and generated population. Physical accounting and
no-early-departure checks passed. Only compact summaries were synchronized.
Preflight t214837 stopped on the native-mask check with zero traffic; corrected
preflight t214838 passed before any path started.

The old development days froze the full V406 reference at 73.596% fixed10 and
26.404% fixed15. The capped reference is 3.873% fixed0 and 96.127% fixed10.
Weights operate on whole-day endpoints, never on commands inside a path.

Candidate minus reference, 95% paired day-bootstrap intervals:

| Contrast | Holding s/trip | Journey s/person | Actual CV2 | Readiness s/trip |
| --- | ---: | ---: | ---: | ---: |
| Full vs frozen mixture | +2.016 [+.471, +3.606] | -9.453 [-11.271, -7.532] | -.008170 [-.010611, -.005768] | -6.525 [-8.106, -5.039] |
| Capped vs frozen mixture | -.536 [-1.170, +.089] | -3.363 [-4.092, -2.606] | -.005258 [-.006957, -.003808] | -3.505 [-4.272, -2.753] |
| Capped vs pure fixed10 | -4.031 [-4.674, -3.397] | -4.437 [-5.224, -3.640] | -.001653 [-.003439, -.000103] | -4.659 [-5.467, -3.868] |
| Full vs capped | +13.863 [+12.713, +15.069] | -1.411 [-3.183, +.297] | -.013567 [-.016032, -.011029] | +3.530 [+2.456, +4.632] |

**Dose first.** The full candidate misses the +/-3 s/trip criterion: its point
estimate is inside, its interval is not. Its four favorable outcome gates
cannot rescue the matched-dose claim. The capped candidate meets the dose
criterion and all four outcome gates. Its public delay is -918.3 s/day
[-1119.3, -721.4] and journey/CV2/readiness each improve on all 16 days versus
its frozen mixture. Waiting contributes -2.185 s/person and onboard time
-1.178. The pure fixed10 contrast also favors capped control, but its dose is
not equivalent; CV2 improves on 11/16 days. Readiness and public delay here
measure the same delay burden at different scales.

**What the high bins buy.** Full V406 versus capped improves waiting by 3.908
s/person and CV2 on all 16 days, but increases onboard time by 2.497 s/person,
readiness delay by 3.530 s/trip and public delay by 924.8 s/day. The journey
interval crosses zero. Access to high bins changes the operating tradeoff;
it is not required for the joint benefit against a simple rule. Full V406
also trades worse CV2 for better journey/readiness against pure fixed15.

The read-only command audit t214895 reconciles all 83,200 commands per
candidate with the physical holding ledger. Relative to native fixed10 on
each candidate's own states, capped V406 increases 2,063 commands and
decreases 4,640, adding 13,920 and removing 34,180 command seconds. Of
6,703 changes, 5,862 use the censored-demand model. Full V406 uses a bin
above 10 s on 3,548 queries. These counts separate real selection from
fallback; the native projections are not separate simulated trajectories.
Capped paths take about 18--34 min/day versus 31--59 for full V406, with
per-day callback p95 0.254--0.888 s versus 0.793--1.237 s. This is not a
one-second deadline guarantee.

**Scope and next step.** The capped controller passes against a fixed daily
mixture with the same action ceiling and comparable mean total holding. This
does not establish equal holding per trip, dominance over all simple rules,
or a learned-upper, timetable, or frequency contribution. In particular
the zero-holding component can change daily trajectories. A within-day
simple comparator should draw a legal 0/5/10 transition conditional only on
the projected native bin and legal mask, with a separate policy random
stream. Match upward and downward moves rather than compare with one-way
thinning. Estimate transitions and calibrate mean holding only on consumed
development days, then freeze before comparison on new days; report realized
bin composition as well as total dose. Do not tune on V408/V409 confirmation
days or alter the frozen gates. Keep capped V406 as a high-demand research
candidate, not the main policy.

[Frozen protocol](PROTOCOL.md) | [Calibration](review/calibration/results.json) |
[Full small aggregate](review/aggregate/results.json) |
[Command audit](review/placement/results.json)
