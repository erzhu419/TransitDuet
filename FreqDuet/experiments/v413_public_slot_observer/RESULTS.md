# V413: Independent Public-Slot Forecast Readout

All eight reserved days (`2530001`--`2530008`) completed once, without retry,
training or policy changes. Every day has 260 advance and 262 at-public-clock
records; two advance slots per day precede simulation time zero. Server-side
schedule and cohort-label audits passed. Paths took 33.1--39.4 minutes.

| First scenario, supported slots | Advance 300 s | At public clock |
| --- | ---: | ---: |
| Samples / all in-range slots | 1928 / 2080 | 1941 / 2096 |
| Correct late-launch classification | 1847 / 1928 (95.80%) | 1941 / 1941 (100%) |
| Observed-ready-only correct, same slots | 1651 / 1928 (85.63%) | 1939 / 1941 (99.90%) |
| Missed late launches / false alarms | 30 / 51 | 0 / 0 |
| Actual-launch clock MAE, seconds | 73.41 | 51.39 |

Advance accuracy improves on all eight days; the mean of daily paired
accuracy differences is +10.16 percentage points. With no earlier unlaunched
slot, 410/447 are correct versus 229/447 for ready-only. With earlier slots
queued, it is 1437/1481 versus 1422/1481, improving on six days and worsening
on two. Thus most incremental value lies outside the existing launch queue.
At-clock classification mainly repeats current readiness, not advance warning.

Among complete inbound cohorts with no bus ready and no earlier slot queued,
advance first-return rank is correct on 378/379 slots. All 379 predicted gaps
exceed 30 s, including the error (predicted gap 62 s, realized order reversed
by 2 s). At clock, 3/865 ranks are wrong: 3/8 at <=30 s, 0/857 above 30 s.
The eight near-ties occur on only two days. The second demand scenario gives
identical classifications and rank results, so it does not reveal these errors.

**Decision.** Retain a positive independent validation of five-minute supply
forecasting, not a control benefit or a certified return-order trigger. Keep
V412's slow-speed failure and the main policy unchanged. The useful next step
is a bounded read-only decomposition of the 30 misses and 51 alarms, separating
queued obligations from return/launch timing error, before proposing a control
change. Do not tune the 30 s threshold or expand training on these days.

[Small server readout](review/readout/results.json)

[Completed 81-case error accounting](ERROR_AUDIT.md)
