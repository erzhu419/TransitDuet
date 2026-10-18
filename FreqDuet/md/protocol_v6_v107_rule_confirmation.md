# V107: frozen quarter-gain rule confirmation

V106 found large passenger/readiness savings at gain 0.25, but did not establish
CV2 nonharm. Keep both forward025 and bilateral025 as co-primary rules; do not
pick a family winner, alter gains, interpolate, fit a CV margin or update actors.
Higher-gain results remain reported V106 development tradeoffs, not new candidates.

Reuse the V106 implementations unchanged. Native B200, V96 preserve and zero
holding are controls; upper weights/deployment background, fleet, clocks,
causal evidence and seven feasible holding bins remain the same. Zero is not
an exact fixed-upper external baseline. Both actors and duals remain frozen.

Use the other ten archived actors (all 12 `TRAIN_SEEDS`, excluding 12037/933083),
and 16 new days 2071001--2071016. They are unseen in V106, not newly trained or
untouched throughout the project's history. Native/preserve retain three paired
samplers; zero and both deterministic rules retain one shared path per actor/day.
Total: 1440 full-day rollouts. No adaptive extension after reading results.

Average samplers before the existing 5000-draw crossed actor/day bootstrap.
Report absolute costs/intervals, paired changes against all three controls,
the direct same-gain family contrast, actor/day axes and evidence coverage.
For each co-primary rule require passenger upper < 0 and CV2, readiness,
unlaunched/incomplete upper <= 0 against BOTH native and preserve. No margins.
The existing two-sided 95% intervals provide 97.5% one-sided upper bounds;
each rule's claim is an intersection of all constraints, with two co-primary
claims. Zero comparisons describe the faster/less-regular tradeoff, not a
required Pareto victory. A crossing-zero interval is inconclusive, not equality.

Smoke: new tests and five exact cached V106 control replays; no new confirmation
day labels. Formal shards wait for this real marker; aggregate waits for every
actual shard. Scheduler on node001--node006 only, dedicated CPU environment;
only markers/final small JSON return locally. A pass does not switch main or
establish a frequency-separation contribution. A failure keeps the measured
tradeoff and ends this fixed confirmation budget, without another gain search.

## Run record

Frozen source `e994b9693a`; V106 control implementations remain unchanged.
Smoke `t183525` passed on node002: six tests and five exact cached replays,
including metrics, action counts, physical contracts and evidence coverage.
No fresh confirmation labels were used. Formal `t183526--t183529` covers
0--300/300--720/720--1020/1020--1440, with 10/14/10/14 workers from the
live CPU planner. Aggregate `t183530` waits for their four actual markers.
All jobs are restricted to node001--node006; only small outputs synchronize.
At dispatch, two 14-worker shards did not fit the remaining CPU capacity.
Resize only queued `t183527/t183529` to 7/9 workers (current total 36);
all ranges, seeds, controllers, source and aggregate dependencies are unchanged.

## Outcome

All 1440 paths and aggregate t183530 finished with exit 0. Only the 165653-byte
final JSON was retrieved. Both rules reduce passenger time and readiness in all
ten actor means and all sixteen day means versus both B200 and preserve.
Paired crossed actor/day 95% intervals, candidate minus control:

| Contrast | Journey min | Actual CV2 | Readiness s |
| --- | --- | --- | --- |
| forward025 - B200 | -1.1953 [-1.3880,-1.0172] | +.001367 [-.004692,.008030] | -442.27 [-513.70,-371.56] |
| bilateral025 - B200 | -1.4031 [-1.5935,-1.2260] | -.006404 [-.011435,-.001376] | -462.18 [-534.71,-390.81] |
| forward025 - preserve | -.9994 [-1.1668,-.8470] | +.004546 [-.001382,.011118] | -339.35 [-393.26,-285.39] |
| bilateral025 - preserve | -1.2072 [-1.3709,-1.0581] | -.003225 [-.007144,.001291] | -359.26 [-413.94,-304.91] |

Bilateral passes the original joint gate versus B200, but neither rule passes
against BOTH controls: actual CV2 versus preserve remains inconclusive. Forward
CV2 worsens in 9/10 actor means and 10/16 day means versus preserve; bilateral
worsens in 1/10 actor means and 4/16 day means. These counts are descriptive,
not independent tests. All arms complete all trips. Bilateral uses follower
evidence in 696319/832000 decisions (83.69%); remaining decisions use fallback.
Relative to zero, forward/bilateral cost +.9194/+.7116 journey min and
+97.97/+78.06 readiness s, while reducing CV2 by .13104/.13881. Bilateral minus
forward improves journey by .2078 min, CV2 by .007771 and readiness by 19.91 s
(all paired intervals exclude zero), a reported co-primary comparison, not a
post-hoc controller selection. The waiting-time benefit versus zero remains
unresolved; the additional passenger cost is chiefly onboard exposure.

Decision: close the fixed confirmation budget without gain tuning, additional
confirmation days or gate relaxation. Keep the verified bilateral-versus-B200
positive result and the preserve/zero tradeoffs; main remains frozen. V108 is
a passive timing diagnosis on consumed paths, not another promotion attempt.
