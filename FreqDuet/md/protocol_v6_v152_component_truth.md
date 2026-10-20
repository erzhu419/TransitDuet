# V152: read-only true-component accounting

Frozen V151 source1b783ea8f3/prior,V146(3s,0s),3600s window,.1/.1 tails.
Read-only saved-clock reconstruction; full MAE=sum(component MAE)-2*cancellation
before existing episode weighting and crossed actor/day statistics. No new gate.
Audit source9fbd08c51b;t199247,node006,8 workers. Superseded t199246 cancelled
during launch after excluding opaque pre-launch clocks from truth partitioning.

## Readout And Decision

t199247 exited0;4 tests passed;192 pairs/384 paths and original endpoints
reconciled. Only224652-byte summary pulled. Paired old/handoff component support
831053/831516 queries;4475/4002 opaque references excluded, not counted as zeros.

- All-day service MAE9.361->9.137s /9.366->9.141s;clearance13.322->8.614s /
  13.353->8.593s. Running and separable holding predictions remain unchanged.
- Hour2(7200--10800s):running bias-29.34s;service bias-0.69->-4.90s. Paired
  full MAE worsens1.116/1.108s,CI[0.604,1.607]/[0.597,1.602]. Service-MAE
  increase contributes0.497/0.489s,reduced cancellation0.619/0.619s;both
  individual CIs cross0. Full-population deltas remain0.786/0.877s.
- Four handoff launches share one context(day2752001,time47120,query259/16/
  reverse,follower261/bus4). True running/service/holding/ticks=475/34/0/8s;
  reference373.527/102.491/35.455/5.564s nearly cancels to truth517s.
  Joint keeps other components,improves mean service to61.500s,but total MAE
  worsens40.918s. Event-only ETA511.945s;joint476.045s. No crossed CI here.

## Next

Keep V151 sealed. Localize running underprediction by segment/time using saved
observable speed and completed-prefix lag before designing new calibration;
do not restore biased service to mask it. No replay,refit,training or policy
change. Summary: results_freqduet/protocol_v6_v151_historical_service/
component_truth_9fbd08c51b/status.json.
