# Stage164: Independent Duration Calibration

Stage163's fleet penalty changes only 8/880 and 17/880 choices. Initial predicted
peaks are 5-6, below the 12-bus target; a later agreement with realized peaks does
not validate the initial prediction. Measure an operational trip-time prior
before changing the selector or expanding learned-policy experiments.

Two roots, five regimes, one fitting and one independent check scene per cell:
20 full frozen episodes / 1,227,600 native ticks. Seeds are
`810000000 + root*1000 + regime_index*10 + split_index`, separate from earlier
evaluation. Execute the unchanged window forecast with zero residual and frozen
native weights, fleet12, 44 upper commitments, 262 trips and the same daily clock.
Record initial current-route freeflow prior at the first callback and actual
completed launch/completion times from the passive ledger. Raw logs/checkpoints
remain remote; output contains only compact timings and physical outcomes.

Pool a trip-weighted direction mean per root from the five fit scenes only.
Compare freeflow and fitted prior errors on the five check scenes, including all
regime panels. Check outcomes never enter the fitted prior. The next closed-loop
experiment will isolate this prior change with matching demand/supply controls;
there is no new allocation-performance claim from these calibration episodes.
Ten unpinned scheduler cells on node001-006, 1 CPU/3 GB each. All 20 episodes and
their clock, fleet, schedule and completed-trip observations qualify.
36 focused tests pass, including held-out exclusion, seed separation, passive
dispatch/goal equivalence and invalid timing/split/budget rejection. Tests use
stub environments; full native calibration runs only through the scheduler.

Run `native_transit_duration_calibration_stage164_frozen_20261011_r1`, source
revision `d0000c25e9`: t142154-t142163 are all done on node005/node006.
Placement was unpinned; the allowed pool remained node001-006.

| Root | Direction | Fitted prior (s) | Freeflow check mean error (s) | Fitted check mean error (s) |
| --- | --- | ---: | ---: | ---: |
| 397 | False | 2046.25 | 1274.65 | 21.68 |
| 397 | True | 2000.85 | 1230.17 | 17.56 |
| 401 | False | 2049.63 | 1271.32 | 26.03 |
| 401 | True | 1990.24 | 1208.95 | 20.29 |

Initial freeflow is 784 s in every cell. The fit uses 655 completed trips per
root/direction; all five check regimes improve on episode-mean duration error.
This identifies a substantial cold-start prior mismatch. Preserve the registered
whole-day fitted mean, without retuning it on checks. Stage165 will replace only
the pre-first-completion duration estimate and measure closed-loop cost, wait,
fleet and nonfleet outcomes with all previous physical rules unchanged.

## Limitations

This is training-scenario duration calibration, not independent confirmation of
Freq-HRL. A direction mean does not model headway-dependent holding, uncertainty,
future speed shocks or capacity; its closed-loop value still needs measurement.
The 18-26 s errors are for episode means, not individual-trip errors. Fitting
standard deviations are 175-193 s. On check scenes the first six departures'
mean differs from the prior by 54-74 s in False direction and 225-285 s in True
direction: a day-level mean is not an exact cold-start or peak-fleet predictor.
