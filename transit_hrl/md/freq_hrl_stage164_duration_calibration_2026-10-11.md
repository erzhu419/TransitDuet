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
Ten unpinned scheduler cells on node001-006, 1 CPU/3 GB each. Results pending.
36 focused tests pass, including held-out exclusion, seed separation, passive
dispatch/goal equivalence and invalid timing/split/budget rejection. Tests use
stub environments; full native calibration runs only through the scheduler.

## Limitations

This is training-scenario duration calibration, not independent confirmation of
Freq-HRL. A direction mean does not model headway-dependent holding, uncertainty,
future speed shocks or capacity; its closed-loop value still needs measurement.
