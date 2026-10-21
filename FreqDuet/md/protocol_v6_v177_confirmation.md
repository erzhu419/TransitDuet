# V177: independent confirmation of frozen V176

Apply source 36108d4f49 once to the 64 V175-reserved demand days
3352001-3352064, samplers 1751000000-1751000063, all 12 actors and both modes:
1,536 passive paths. No map updates, refitting or changes to the controller.

Report coverage, both tails, finite fraction, interval scores and widths versus
coarse/V167/V172 on the same candidate-finite query mask. Retain all six scopes,
especially clearance and hour13 moving-launch. Missing finite labels/references
remain inconclusive; native/low/unseen support stays abstained. Preserve query
snapshots and labels separately, writing interval predictions before attaching
labels. The existing full conjunction remains the only admission rule.

Use equal query/episode/actor/day reduction and the fixed 5000-draw shared-day
crossed bootstrap. Save actor/day marginals and score/width/tail decompositions
on the server. Submit 96 eight-pair shards plus a receipt-dependent aggregate,
all high priority, manually dispatched to node001-node006. Pull only tiny shard
receipts and the small final summary, never raw trajectories or checkpoints.

## Launch record

Source d1454cfc39 passed all nine server-side preflight tests (t210496),
including reserved-day isolation, frozen-map geometry and runtime equivalence.
Collection tasks t210498-t210593 and aggregate t210594 were submitted at high
priority and explicitly manually dispatched. At launch verification, 88 shards
were running and eight awaited resources; no failures or retries were present.
Server logs contained completed V177 confirmation paths. The aggregate waits
for all 96 tiny shard receipts. This is launch evidence, not a coverage verdict.

## Independent results

All 1,536 paths/96 shards completed. Aggregate t210594 and read-only t210600
ran high/manual on node005; all 12 grids have complete finite labels/references.
Retrieved only 22,916 bytes of receipts and 249,700 bytes of summaries. All
frozen items remain unchanged; crossed inference uses 12 actors/64 days/5,000 draws.

Point means in percent; tails/coverage condition on the same finite mask:

| Scope | Mode | Coverage | Lower miss | Upper miss | Finite fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| Clearance | old | 84.703 | 9.250 | 6.047 | 89.054 |
| Clearance | handoff | 84.745 | 9.245 | 6.010 | 88.765 |
| Hour13 moving-launch | old | 84.946 | 8.939 | 6.115 | 98.559 |
| Hour13 moving-launch | handoff | 84.900 | 9.031 | 6.069 | 98.695 |

Full gate: **not admitted**. All coverage/support gates pass; all/active pass
fully. Target upper-tail upper95 is below 7%. Clearance fails lower-tail
upper95 (10.214%/10.231%) and strict coarse-score improvement. Hour2 fails
upper-tail upper95 (10.566%/10.575%, means 9.908%/9.910%). Hour13 and moving-launch
fail coarse-score non-harm. These are one-sided 95% gate bounds.

Paired deltas in seconds, candidate minus reference; score lower is better.
Widths are versus coarse/V167/V172; score CI is two-sided 95% crossed:

| Scope/mode | Score vs coarse | Score vs V167 | Score vs V172 [CI] | Width deltas |
| --- | --- | --- | --- | ---: |
| Clearance/old | +1.795 | +4.691 | +4.691 [2.901,6.404] | +14.638/+15.087/+15.087 |
| Clearance/handoff | +2.208 | +5.082 | +5.082 [3.296,6.772] | +14.911/+15.474/+15.474 |
| Moving/old | +0.973 | +4.745 | +4.676 [2.745,6.504] | +12.969/+16.942/+16.649 |
| Moving/handoff | +1.384 | +5.206 | +5.055 [3.148,6.872] | +13.103/+17.162/+16.950 |

V167/V172 score deterioration is significant; coarse-score improvement is not
confirmed. Upper expansion repairs 3.43/3.47 percentage points of clearance
misses and 3.71/3.71 moving-launch points. Upper-penalty savings of 10.40/10.39s
and 11.97/11.89s cannot offset width costs. Lower penalties stay unchanged.
All 12 actor means worsen versus V172; day means worsen on 51/52 clearance
days and 46/47 moving-launch days. Clearance lower misses exceed 10% on 23 days
per mode (descriptive, not an additional gate).

Whole-day scores beat coarse (-6.199/-6.181s) and V167 (-3.487/-3.470s) for all
12 actors/64 day marginals, but worsen versus V172 by +0.149 [0.098,0.198]/+0.161
[0.111,0.209]s. Hour2 is exactly V172: inherited score gains, unconfirmed tail bound.

Decision: end V176 promotion; retain its upper-tail evidence separately. All 64
confirmation days are consumed. Next: read-only localization of day-shared lower
errors and excess-width costs by observable state/support. No fitting, new
collection, gate changes or policy switch.
