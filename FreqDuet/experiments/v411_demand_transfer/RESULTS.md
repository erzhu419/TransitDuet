# V411: Frozen Transfer Passes at 1.35x Demand

All 48 paths on 16 new days completed without retries. The first six paths
passed the execution-only release gate and were reused as formal cells.
Each arm used the declared demand multiplier and 12 vehicles. The original
physical, no-early-departure, and 5,200-event pairing checks passed. V410's
controller, transition table, tilt, and policy seed were not changed.

Capped V406 minus the frozen within-day rule, paired 95% day-bootstrap
intervals. The two regimes are not pooled.

| Metric | Nominal 1.0x | Stress 1.35x |
| --- | ---: | ---: |
| Holding, s/trip | +2.118 [+1.195, +3.022] | -1.546 [-2.192, -0.961] |
| Journey, s/person | -3.355 [-3.822, -2.732] | -5.447 [-8.897, -2.896] |
| Actual CV2 | -0.011172 [-0.013632, -0.008719] | -0.010464 [-0.014114, -0.007315] |
| Readiness, s/trip | -1.437 [-2.523, -0.391] | -3.147 [-4.409, -1.915] |
| Public delay, s/day | -376.4 [-661.1, -102.5] | -824.6 [-1155.1, -501.6] |

The three bin-share gates and all outcome gates pass in both regimes.
The nominal holding interval exceeds the +3 s/trip margin by 0.022 s/trip;
retain the joint failure. This is a failure to establish equivalence under
the protocol, not a test showing the mean difference exceeds +3.
Nominal journey and actual CV2 improve on 8/8 days, readiness on 6/8.
Stress passes the dose and all other gates, with all three outcomes
improving on 8/8 days. Its journey gain is -4.333 s/person of waiting and
-1.114 s/person onboard; the unboarded-count interval crosses zero.

Against pure fixed10, capped V406 reduces journey by 3.929/6.103 s/person
and readiness delay by 3.403/4.821 s/trip at nominal/stress. CV2 is
-0.002001 at nominal (interval crosses zero) and -0.003151 at stress
(interval below zero, 8/8 days better). Both pure-rule comparisons use
less holding and are not matched-dose tests. The randomized rule alone
worsens CV2 against fixed10 on all eight days in each regime.

**Decision.** Preserve V410 and the new matched-dose result at 1.35x.
The tested high-demand benefit transfers to a second pressure setting,
not to every load at comparable dose. One corridor, one fleet, and eight
days per new setting do not establish route transfer or a frequency
contribution. Do not recalibrate on these days, round away the nominal
failure, or change the main policy. The next useful evaluation is a
separate route/operating-condition transfer with an independent
calibration split if it requires a new dose reference.

[Frozen protocol](PROTOCOL.md) | [Engineering gate](review/engineering/ready.json) |
[Small aggregate](review/aggregate/results.json)
