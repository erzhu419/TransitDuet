# V153: read-only running-time localization

Frozen V151/prior,main,V146(3s,0s),3600s window,.1/.1 tails. Source cf7e297449,
t204485,node002,8 workers,exit0,no retries;4 tests passed.192 pairs/384 saved
paths reconciled with V152 and exact segment budgets. Only535019-byte summary
pulled;no replay,fitting,training or frozen-item change. Superseded t204484
cancelled while queued. Comparisons use strictly completed events,not future inputs.

## Readout And Next

Hour2 old running bias-29.373s,CI[-32.039,-26.419];all12 actor and16 day margins
negative,handoff similar. Largest budget contributions(seconds per weighted
query):reverse14->13 -4.976,10->9 -4.578,7->6 -3.742,17->16 -2.912,3->2 -2.900;
forward4->5 -2.776. These are globally normalized contributions,not conditional
per-segment errors. Mean sample age1807s,latest173s;52.4% entered before hour2.

On51348/51354 old/handoff queries with same-hour completed support for every
remaining leg,native bias-25.825/-25.849s versus same-hour-0.880/-0.913s.
MAE delta-4.405/-4.433s,CI[-5.663,-3.209]/[-5.713,-3.243]. However clearance
MAE worsens3.866/3.874s,CI[2.178,5.415]/[2.206,5.410]. Naively extrapolating
observed moving-time averages across all remaining legs worsens hour2 MAE
16.831/16.853s;current-segment averages worsen120.012/120.854s.

Four handoff launch regressions share one context,not four independent events.
Reverse19->18 predicts108.2s versus165s(-56.8);17->16 predicts72.273s versus
126s(-53.727). Other legs offset part of their error,total-101.473s. These two
legs already have one hour13 completion each(85s/58s),164s/14s old;using those
alone would worsen underprediction. At launch,observed progress/running time
are zero;3.5m/s is a route-speed fallback,not measured follower moving speed.

Decision:keep V151 sealed. Next design a segment/time running prior with
support-aware shrinkage and uncertainty,using existing calibration-day archives;
preregister clearance/low-support fallback and hour2/full ETA non-harm for
independent confirmation. No hard hourly reset,global speed multiplier or tuning on these
confirmation days. V153 has not fitted or validated a new predictor.

Summary:results_freqduet/protocol_v6_v151_historical_service/
running_lag_cf7e297449/status.json. Conditional support and sparse-launch limits
above remain;prefix mixing is a diagnostic association,not unique causal proof.
