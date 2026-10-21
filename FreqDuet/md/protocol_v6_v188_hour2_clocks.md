# V188: hour2 running reversal, existing calibration archives only

Original64 days2951001-2951064,12 actors,old/handoff. Read V166 query/residual
packets and saved arrival/departure/launch arrays. First reproduce V187 hour2
endpoints and weighted moments,including its strict first-query completed prefix.
No replay,training,point/interval fitting,new confirmation or V185 outcome use.
V151/V146/V172,window3600s,tails and main policy stay frozen.

Reconstruct current running predictions from recorded per-segment CompletedMeans
inputs,partial first-leg geometry,and strict arrival-completion clocks. Contrast
those clocks with residual availability at target ready; retain missing running
legs explicitly via reported analysis mass. Clock/phase facts use only completed
prefixes and query observables; current outcomes are diagnostic labels only.

Fixed V180 geometry/time strata,then fixed quarter-hours within hour2. All64,
first32,last32;5000 crossed actor/shared-day draws,seed20260803. Controls:service
prefix;query time/distance/horizon;completed-prefix clocks and stages;current
running forecast level. Exact covariance identity:error=truth-prediction.
Partial associations describe sources,not causal effects of removing a predictor
input. No sign inversion,threshold selection or mapping release from this audit.

High/manual scheduler,node001-node006. Detailed cells stay server-side;retrieve
only small summary and logs. Next decision follows replication and decomposition.
