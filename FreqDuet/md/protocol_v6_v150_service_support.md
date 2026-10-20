# V150: read-only V149 support and fallback diagnosis

V149 archives only. D7996d60b4 points,V146(3s,0s),3600s window,.1/.1 tails and
main policy unchanged. No new predictions, replay, fitting or training.

## Readout

Scheduler t199127(45a6b4317f),t199132(e840658090): node006,exit0,12 tests passed,
384 actor/day pairs,768 paths; saved points/support/component totals reproduced.

- No clearance per-stop forecast has9+ completed same-phase labels; supported
  forecasts average2.23. Accuracy with larger support is unknown.
- Remaining downstream-service bias is11.904/11.930s(old/handoff). Per-stop
  episode-weighted contributions: no same-phase samples3.484/3.475s;1--8 samples
  8.420/8.441s; retained boundary0/.015s. These are allocated contributions,
  not conditional errors; conditional biases are4.180/4.168 and2.856/2.860s.
- Of6921/6909 service-survivor fallbacks,5579/5572 have9+ historical durations
  but none survives elapsed service age. This is tail depletion, not no history.
- Ready-file insertion order proves134/147 already-observed same-clock ready
  events;1240/1202 occur later in that tick;5547/5560 occur after query time.
  Only about2% qualify for an already-observed-ready state correction.
- All899/909 zero-position launch boundaries have the required component means.
  Runtime launches before bus updates; handoff logs the observation, but the
  generic origin bridge misses it. Own-bus query_load is not follower APC;
  no service fallback has an earlier current-stop follower query_load.

Small JSONs: results_freqduet/protocol_v6_v150_service_support/{summary_45a6b4317f/
status.json,order_e840658090/ready_order.json}. Original V149 gate remains false.
Existing5000-draw crossed actor/day bootstrap; incomplete groups have no CI.
Nine is descriptive, not an accuracy guarantee or a causal sample-size effect.

## Next

Design an independent-day phase-service prior and elapsed-age conditional tail,
retaining unsupported-tail abstention. Bridge only actually observed launch/ready;
historical labels remain strictly<query. Pre-register the estimator and separate
new calibration/confirmation days; seal the fitted prior before confirmation.
Do not fit or tune on V149. No predictor changed here; no large data pulled.
