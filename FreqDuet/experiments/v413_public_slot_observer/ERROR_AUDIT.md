# V413: Advance Error Accounting

Read-only analysis exactly reproduces the 30 misses and 51 false alarms in
the frozen first-scenario, 300 s readout. No new traffic, training or changes
to the policy, thresholds or historical gates.

| Error subset | Slots | Earlier obligations | Same vehicle/cycle | Mean return forecast minus realized |
| --- | ---: | ---: | ---: | ---: |
| Missed late launch | 30 | 23 | 30 | -126.87 s |
| False late alarm | 51 | 21 | 51 | +76.25 s |

All 81 assigned buses and current supplying trips match the realized path.
After earlier obligations consume their supply, the target's required return
ordinal reproduces modeled and actual launch clocks exactly.

All 30 missed launches occur one tick after return, with mean actual lateness
92 s. In 21 of them, some vehicle was available by the public clock, but the
target's required vehicle was not: first-return readiness is insufficient.
All 51 alarms forecast return-binding launches, yet the actual launch was
on time, with mean forecast lateness 42.90 s.

**Interpretation and next step.** This is accounting, not causal attribution.
The observer projects capped bilateral continuation; actual V406 uses
state-dependent choices. The return difference is not pure road ETA error.
Slot receipts lack predicted run/service/remaining-H components. Establish
those matched-cycle components before a model correction or new calibration,
not another rank/queue guard or a uniform ETA adjustment on consumed days.

Synthetic fixture failure `t215000` is retained; corrected task `t215001`
produced this result. All 81 detailed cases remain server-side.
[Summary](review/error_audit02/results.json)
