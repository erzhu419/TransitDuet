# V147: read-only full follower ETA and identity propagation

Freeze V146 dL=3s,dU=0s; V140/V141 point formulas,3600s/all-past window,
rank tails,actors/critic/dual and deployed controls remain unchanged. Read all
768 V146 confirmation paths(12 actors x32 days x old/handoff),not calibration
paths. No replay,training,new point prediction,interval construction or fitting.

Score saved native/V140/V141 gaps against the ready label of the exact predicted
trip AND bus at the target stop. Keep invalid evidence,missing/past labels and
missing forecasts explicit. Group fallback,active/clearance,hour13 and native
finite/abstaining service/ready requests. Average target episodes before paths;
use the existing5000-draw actor/day crossed bootstrap on complete grids only.

Match own trip/stop across modes; distinguish follower identity changes from
forecast-source/status changes. Compare predicted gap change with realized gap
change. Where both cross-identity labels exist,split truth change symmetrically:
average both fixed-identity clock responses,then average both identity replacements.
Cross labels and actual next-departure identity are retrospective targets only.
For bilateral requests,recover executed holding change as clipped physical-ready,
identity,prediction-error,forward-gap parts plus quantization/execution remainder.
Missing anchors are not imputed; these identities are accounting,not mediation.

Tests,one saved-pair preflight,all-path audit and statistics run through scheduler
on node001--006 in the existing independent CPU environment. Details stay remote;
only the final small summary is fetched. Consumed-day diagnostics cannot promote
the controller. V146 stage-duration coverage is not full-ETA interval coverage.
Next decision depends on full-gap accuracy,identity/source-change error and
their actual holding accounting; no retuning on these days.

## Results And Decision

Source1c5ead2597,t199001/node006 exit0;10 tests and saved-pair preflight pass,
8 workers,all384 pairs complete. Only97943-byte summary fetched;raw data stay remote.
Episode-first full same-vehicle ETA MAE(s),native -> frozen V141:

| Scope | old | handoff |
| --- | --- | --- |
| All | 50.331 -> 45.661 | 50.222 -> 45.553 |
| Original fallback | 66.086 -> 45.022 | 66.074 -> 45.023 |
| Clearance | 44.865 -> 40.925 | 44.862 -> 40.947 |

All-day paired changes -4.670[-4.847,-4.493]/-4.669[-4.840,-4.495]s;
all12 actor/all32 day margins improve in both modes. Matched-path gap-change
MAE2.294->1.267s,delta -1.027[-1.279,-0.812];all12 actors/all32 days improve.
All1673236 matched valid decisions retain the same follower trip/bus across modes;
each mode's valid labels also match actual next-departure identity. No identity
replacement effect is estimable from this cohort.41759 source/status-switch
decisions occur in349/384 cells:conditional change-MAE53.694->20.312s,
holding delta -0.533s/request,forecast-error accounting -0.717s,physical-ready
+0.049s,forward +0.096s,execution remainder +0.040s;identity zero. These are
conditional descriptive means,no subset crossed CI or action-level mediation.

Remaining issue:clearance full-ETA bias +21.540[18.291,24.889]/
+21.547[18.297,24.933]s,positive for every actor/day margin;all-day bias is
only -0.294/-0.418s. Current-stage interval calibration did not remove full-ETA
timing error. Ready-abstain groups contribute381/380 cells,so their CI remains
inconclusive. Main,points,window,tails and V146 offsets stay frozen.

Next: retain both calibration and point-prediction gains;no control admission.
Read-only component accounting should isolate clearance full-gap overprediction
and remaining source-switch errors into current-stage and downstream running,
service,holding/execution clocks. Do not add a global point offset or tune on
these consumed days;a later repair needs a separate design and confirmation.
