# V406 Corrected Development Result

The implementation and bounded server comparison are complete. None of the
three candidates passes the frozen joint screen. Preserve the single-action
controller's positive service/regularity signal; do not promote the paired
or coordinated method or extend its confirmation matrix.

Corrected single-action controller minus each comparator; two paired demand
days per condition. Negative means improvement.

| Comparator | Condition | Journey s/person | Actual CV2 | Readiness s/trip | Public delay s/day |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed, always-active 10-s rule | Nominal | -1.366 | -0.009831 | +4.198 | +1100.0 |
| Fixed, always-active 10-s rule | Heavy | -3.588 | -0.011079 | -0.737 | -193.0 |
| Legacy dynamic-plan cap_10 | Nominal | -10.023 | -0.000519 | -7.693 | -2104.5 |
| Legacy dynamic-plan cap_10 | Heavy | -11.544 | +0.007064 | -12.254 | -3302.0 |

Journey and actual CV2 improve on all four days against the always-active rule.
Nominal readiness worsens on both days; heavy readiness worsens on one day.
Against the legacy cap, heavy CV2 worsens on both days and nominal CV2 on one.
This is an operational tradeoff, not solely an uncertainty or small-sample issue.
The always-active rule itself saves 8.657/7.956 s/person versus the legacy cap,
while worsening CV2; those savings are not the model's independent contribution.

Adding paired transfers to single-action control worsens mean journey by
1.500 s/person nominal and 2.136 heavy. Adding release coordination to transfers
leaves all nominal endpoints unchanged; heavy journey increases 0.183 s/person
and readiness 0.260 s/trip. One release edit is published on heavy day 2460202;
the other three joint paths publish none. No stable incremental coordination gain.

The single-action controller changes 799/755 nominal and 631/596 heavy queries
relative to the clipped reference. Heavy changes occur in censored-clearance
states. These counts exclude mandatory clipping of the uncapped native rule.
There is real model control, not merely an inactive controller or a new cap.

Twenty admitted paths: twelve corrected candidate runs plus eight unaffected
controls reused from run01. All paired 5,200-event streams and physical accounts
pass. Candidate full-day wall time is 29--64 minutes on the CPU servers. No
candidate early departure occurs; all original vehicles and trips are retained.

## Limitations

Four consumed development days, no independent future replication or training.
Heavy paths retain 421--512 unboarded passengers across the five arms; their
waiting is included to the fixed evaluation endpoint. This is not full demand
clearance. There is no learned-upper, APC-cost ablation or LF/HF contribution test.
Forecast nonharm at each query does not establish whole-day nonharm. These
different controllers/timings are not an equivalence-preserving speed benchmark.

Evidence: review/unit_readout/results.json and review/unit_checks/results.json.
The incorrect-unit run01 readout and V405's two late audit failures stay archived;
they are not substituted for corrected scores. Total charged traffic this turn:
V405 20 + V406 original 20 + unit repair 12 = 52. Zero automatic traffic retries.
Original TransitDuet, frozen paper results and main policy remain unchanged.
