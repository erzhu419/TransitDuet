# V174: read-only V173 upper-tail diagnosis

Use only the 768 saved V173 paths on consumed demand days 3152001-3152032.
Analyze clearance and hour13 moving-launch against the coarse and V167
references on the exact V173 candidate-finite mask. Use exact saved labels.

First reproduce per-path V173 endpoints and crossed intervals. Then report
upper-bound contraction/expansion, new/repaired upper misses, exceedance size,
and the width/lower-tail/upper-tail contributions to interval-score changes.
Keep equal-query/episode/actor/day weights and the shared demand-day bootstrap.
Partition actor/day/interaction variation descriptively; do not interpret it
as causal variance or use these consumed days to select a replacement map.

All processing and tests run through scheduler, explicit high priority and
manual dispatch, node001-node006 only. Pull only the small summary. No fitting,
new replay, collection, training, frozen-item changes or policy promotion.

## Result

Task t210451, high/manual, node005, source caeaad8fed: 7 tests passed;
768 paths and all four 384-cell grids reproduced V173 endpoints and crossed
intervals. Only the 132,198-byte summary was fetched locally.

Numbers below are old / handoff; probability differences are percentage points.

| Comparison | Mean upper-bound change, s | Upper-miss change, pp |
| --- | --- | --- |
| Clearance vs coarse | +0.013 / -0.115 | +0.753 / +0.840 |
| Clearance vs V167 | 0 / 0 (all bounds identical) | 0 / 0 |
| Hour13 moving-launch vs coarse | -3.030 / -3.030 | +1.406 / +1.448 |
| Hour13 moving-launch vs V167 | +0.629 / +0.569 | -0.109 / -0.033 |

Against coarse, clearance creates 2.354/2.437 pp new misses and repairs
1.601/1.597 pp; moving-launch creates 2.715/2.796 pp and repairs 1.310/1.349 pp.
Both scopes' paired upper-miss increases exclude zero in both modes. The
clearance problem is inherited from V167, not introduced by temporal splitting.
Its nearly unchanged mean upper bound conceals harmful conditional allocation.

Against V167, moving-launch tail differences include zero, while score losses
are +0.470 [0.136,0.815] / +0.693 [0.327,1.095] seconds. Lower-tail penalties
increase by +0.387/+0.504 seconds after the lower bound rises +0.452/+0.512
seconds; upper-tail penalty changes include zero. This is not evidence of
systematic upper-bound contraction relative to V167.

Descriptive actor/day/interaction decomposition assigns 93.98%-94.37% of upper-
miss cell variance to shared demand days, versus 0.06%-0.11% to actor means.
Moving-launch day-mean miss rates span 2.57%-18.32% / 2.87%-18.17%; point-bias
variation is also 97%-98% day-associated. These are descriptive partitions,
not causal effects or proof of a calibration-to-confirmation distribution shift.

Next: retain the hour2 finding but stop full-map promotion. Any new tail method
needs a separately frozen calibration-only design and fresh confirmation days;
do not widen bounds, choose keys or refit on these 32 consumed days. No new
experiment was started and all frozen items and the main policy remain unchanged.
