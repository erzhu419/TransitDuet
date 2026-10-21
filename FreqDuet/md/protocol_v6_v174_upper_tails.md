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
