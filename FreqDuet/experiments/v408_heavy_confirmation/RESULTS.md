# V408: Heavy-Demand Confirmation Passed

All 48 paths on 16 new demand days completed, with no retries or training.
Every day passed physical accounting, the generated-population check and the
5,200-event paired-randomness check. All arms used the same fixed public
timetable and 12 vehicles; none departed early. The candidate was the frozen
corrected V406 single-action controller, not V407 or a new main policy.

V406 minus comparator, equal-weight paired-day means with 95% bootstrap
intervals. Negative is lower. The primary comparator is the frozen 60:40
whole-day mixture of the 10 s and 15 s rules; its endpoints are daily expectations.

| Comparator | Journey (s/person) | Actual CV2 | Readiness (s/trip) | Public delay (s/day) |
| --- | ---: | ---: | ---: | ---: |
| Frozen mixture | -11.738 [-14.766, -8.605] | -0.005578 [-0.007712, -0.003326] | -8.376 [-10.349, -6.499] | -2194.5 [-2711.6, -1702.8] |
| Fixed10 | -7.174 [-10.675, -4.252] | -0.013472 [-0.015528, -0.011459] | -0.881 [-2.529, +0.797] | -230.9 [-662.5, +208.8] |
| Fixed15 | -18.585 [-22.488, -14.300] | +0.006263 [+0.003213, +0.009469] | -19.618 [-22.620, -16.813] | -5140.0 [-5926.4, -4405.1] |

All four registered gates pass. Against the mixture, journey and CV2 each
improve on 15/16 days, and both delay metrics improve on 16/16. Public delay
and readiness here reflect the same delay burden at different scales. The
journey exception is day 2480210 (+1.671 s/person); the CV2 exception is day
2480214 (+0.004814). The joint gate concerns mean effects, not every day. The
journey gain splits into 8.434 s less waiting and 3.304 s less onboard time
per passenger. Restricted journey P95 falls 18.806 s [-37.130, -0.387];
the unboarded count difference is -0.125 [-3.413, +3.512].

The comparisons rule out an explanation based only on reducing aggregate
holding. Relative to fixed10, V406 holds 8.693 s/trip more and saves 7.437
s/person of waiting; onboard time changes by +0.263 s/person with an interval
crossing zero. Relative to fixed15, it holds 19.986 s/trip less and saves both
waiting and onboard time, but gives up regularity. Neither contrast isolates
the causal contribution of holding placement at equal total holding.

The read-only command audit aligns all 83,200 candidate queries with executed
commands and the holding ledger. On candidate states, 4,705 commands are longer
than the fixed10 projection and 5,642 are shorter (+69,390 / -41,510 command
seconds); 3,485 commands exceed 10 s. Of the 10,347 changes, 8,230 occur with
the censored-demand model. All 9,299 unsupported or incomplete-reference
queries retain the cap projection. This identifies active selection versus
fallback; a same-state projection is not another executed traffic path.

**Decision.** Retain this as independently confirmed conditional evidence
for the fast controller, not a replacement for the V400--V404 budget study
or the frozen experiment mainline. The next method question is whether the
advantage persists at a comparable holding dose. Calibrate a simple rule
using only consumed development days, then freeze it before any new
evaluation. Separate the ability to exceed 10 s from the value of selecting
where to hold. Do not tune caps or mixture weights on these confirmation days.

**Scope.** One corridor, one heavy-demand setting (1.2x), fixed timetable
and fleet. The result beats the prespecified mixture at its tested operating
point; it is not dominance over every simple rule, an exact dose match, or
evidence for learned upper control or frequency separation. V406/V407's
original development decisions are unchanged. Candidate paths take about
30--59 min/day; callback p95 ranges from 0.748 to 1.228 s across days.
The study does not establish a one-second realtime bound.

[Frozen protocol](PROTOCOL.md) | [Server aggregate](review/aggregate/results.json) |
[Command audit](review/placement_file/results.json)

Aggregation t214814 and read-only command audit t214828 ran on node004.
The initial audit dispatch t214826 exceeded the SSH argument limit and was
cancelled while queued; the file-backed audit adds zero traffic starts.
