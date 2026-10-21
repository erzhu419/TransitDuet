# V170: frozen key time mixtures and hour13 tail composition

Scan `10022dba3a` / `t209953`, exposure report `e5ee4527a4` / `t209960`:
high/manual on node001, 24 actor/mode buckets, seven tests passed at both entries.
V169 hour2/hour13 means and upper-miss partitions reproduced. V167 unchanged;
no replay, training, interval fit or policy change. Only 125,336-byte summary fetched.

Hour2 candidate-key mixture diagnostics, weighted toward actual hour2 matched requests:

| Mode | Hour2 calibration mass | Pooled bias / hour2 bias (s) | Pooled / hour2 upper miss |
| --- | ---: | ---: | ---: |
| old | .072431 | -2.688 / -35.067 | .093424 / .214740 |
| handoff | .072430 | -2.682 / -35.076 | .093412 / .214288 |

Original V167 day/episode/query mixture weights are retained; no quantiles recomputed. The same hour2-relevant keys pool hours0-13, and their average support is approximately 63.79 days. The coarse reference repeats the pattern (hour2 mass .0701; pooled miss .09375 versus hour2 .2155). Thus plentiful pooled support and acceptable pooled tails conceal a time-conditional calibration failure.

Hour13 all matched requests, new upper misses and repaired upper misses are entirely in frozen predicted band [0,900] in both cohorts/modes. This is not a shift between the existing horizon bands. Old-mode new/repaired mean predicted times are 314.9/326.8s on confirmation, versus 324.9/317.4s on calibration. Moving-launch exposure is .9183 -> .9220, but its within-category new-miss rate rises .02244 -> .03018 while repaired-miss rate stays .01989 -> .01975; handoff repeats this pattern. Moving-launch accounts for 94.19%/93.43% of confirmation new misses.

Both modes' leading confirmation new-miss signature is clearance/clearance, direction=True, target_stop=19, structured/moving-launch/not_applicable/[0,900], with 64 calibration days and approximately 27s upper contraction. Leading repaired signature is direction=False/target_stop=7 with approximately 34-35s expansion. Conditional refinement has both useful and harmful cases. Net added upper misses remain +0.893/+0.932 percentage points; descriptive crossed 95% intervals are [+0.134,+1.682]/[+0.176,+1.687]. V168 non-admission remains unchanged.

Limitations: calibration is in-sample; event contributions use the full matched actor/day population, unlike V169's nonempty-cell conditional means. No between-band shift does not exclude within-band duration effects. Key rankings and crossed intervals are descriptive, not new selection or admission rules; residence is not pure holding.

Next: preregister a temporal-tail-calibration comparison against unchanged V167/coarse, checking time-stratified support first and isolating moving-launch tail transfer. Develop on calibration data and reserve untouched confirmation days; no collection started. Keep current point predictions, window, tail levels and controller frozen.
