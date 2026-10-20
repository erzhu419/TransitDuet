# Stage301: Frozen Joint Versus Service-Only

Run: `native_transit_service_comparison_stage301_20261020_r1`.
All 3,524 E32 tasks completed done/exit0, zero retries: 105,952 native
episodes / 28,192 unique state-futures on 400 fresh prefixes. Report
t208473/node006 qualified all receipts, seeds, identities and cost components;
only 17,705 report bytes were pulled. Policies and matching were frozen.

The registered critic-increment primary fails. Joint cost/wait versus raw
passes again, as does the 50% base-wait-retention floor. Joint also beats
matched thinning on both metrics and both CIs, but has no confirmed extra
cost value over service-only or same-count service ranking.

All estimates and endpoints below are multiplied by 1e6. Waiting uses minutes;
negative differences favor the first controller. Prefix CI is stratified
400-prefix bootstrap5000; conditional MC CI uses shared-future differences.

| Comparison | Metric | Mean | Prefix CI95 | Conditional MC CI95 |
| --- | --- | ---: | --- | --- |
| joint - raw | cost | -307.139 | [-403.590, -216.799] | [-376.580, -237.698] |
| joint - raw | wait | -1070.902 | [-1583.164, -673.425] | [-1185.750, -956.054] |
| joint - service-only | cost | 10.981 | [-38.466, 60.105] | [-26.841, 48.802] |
| joint - service-only | wait | 159.565 | [82.814, 250.561] | [110.377, 208.754] |
| joint - matched thinning | cost | -53.776 | [-85.841, -24.428] | [-75.596, -31.956] |
| joint - matched thinning | wait | -200.207 | [-336.754, -92.743] | [-243.304, -157.111] |
| joint - service ranking | cost | -12.734 | [-41.211, 14.901] | [-27.738, 2.269] |
| joint - service ranking | wait | 18.807 | [-4.652, 48.734] | [9.511, 28.103] |
| service-only - raw | cost | -318.120 | [-424.992, -215.424] | [-396.701, -239.539] |
| service-only - raw | wait | -1230.467 | [-1757.211, -825.834] | [-1357.549, -1103.385] |

| Comparison | First16 Cost | Last16 Cost | First16 Wait | Last16 Wait |
| --- | ---: | ---: | ---: | ---: |
| joint - raw | -277.175 | -337.104 | -984.478 | -1157.326 |
| joint - service-only | -3.884 | 25.845 | 156.987 | 162.144 |
| joint - matched thinning | -68.333 | -39.220 | -178.492 | -221.923 |
| joint - service ranking | -13.546 | -11.922 | 21.823 | 15.791 |

Joint retains 94.42% of base waiting gain pooled, 94.80% first16 and 94.10%
last16. Relative to service-only, its service cost worsens by 15.999 (both
CIs above zero), while fleet improves only 5.018 (both CIs cross zero).
The uncertain fleet compensation does not establish an extra total-cost
benefit. Better-than-thinning selection is not proof of learned-critic value.

Stage302 r2 t208490/node005 completed done/exit0, pulling 67,421 scalar bytes
without new fits or simulation. The paired waiting-floor margin
`W_joint - 0.5 W_base` is -503.815: prefix CI [-763.637, -295.120], MC CI
[-573.509, -434.121] in the scaled units. Both E16 margins also have both
CI upper bounds below zero. Each half's raw-relative cost/wait has both CIs
below zero for joint and service-only. Joint versus service-only cost crosses
zero in both halves; waiting is worse with both CIs above zero in each half.
Joint versus thinning last16 cost has prefix CI [-80.581, 1.491], despite
negative MC CI and point estimate; do not claim both-half two-CI cost support.
Service-ranking cost crosses zero in both halves; its waiting difference has
positive MC CIs but crossing prefix CIs. Both controllers' ten group means
favor cost/wait improvement; all fixed-stratum leave-one-prefix point ranges
remain negative. These group and leave-one analyses are descriptive.

Next: stop expanding this frozen critic calibration branch. Stage302 adds
read-only paired half/floor diagnostics; its first run t208475 failed on a
missing direct comparison key. The fix derives differences from the official
versus-raw means, with a regression test and the failure receipt retained.
Stage303 t208491-t208520 is a 30-task, 4,590-episode native software preflight:
raw/service-only/joint use their own executed history and rebuild P8/S8 at
the next decision. Profile0 and all ten pilot scenes are excluded from
performance claims; raw pilots also require a nontrivial-history identity.
The dispatch snapshot has 7 done/exit0 and 23 running, no failures. Raw pilot
t208491/node001 qualified five native episodes including a changed-history
identity; only its 214-byte scalar witness was pulled. Native service worker
t208492/node001 is alive at 99.3% CPU / 608 MiB. The eight-module test suite
passes 55 tests. Full pilot qualification remains pending before formal
closed-loop performance sampling.

Scope: fixed lower, single intervention, 16 fixed historical profiles rather
than independent training seeds. Joint's all-state mean wait improvement is
about 0.064 seconds. Service ranking is a cohort-budget benchmark, not an
online controller. These results do not establish full end-to-end Freq-HRL.
