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
