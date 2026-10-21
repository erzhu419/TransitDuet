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

Completed:source80cdc761f7,t210804,high/manual node004;5 tests and1536 path
checks passed. V187 moments/endpoints and native means/counts/clocks reproduced;
running analysis mass1.0. Pulled122595-byte summary only. Residual-prefix queries
average2205.6s old,completion age1821.8s,label delay383.8s;ready-after-arrival only
12.95s.61.1% of prefix queries precede hour2;52.7% of native samples entered earlier.
Across fixed quarters,true running336.6->294.8s while forecast278.0->296.5s.
Old service-controlled covariance(error)=-178.86=-45.41(truth)-133.45(prediction).
Forecast-level control alone leaves r=-.216,so absorption is not the sole source.
Time/distance/horizon controls reduce running-error r to+.012;adding completed
clocks/stages gives+.001,CI[-.023,.026],upper-miss/score CIs cross0. Both modes
and predetermined halves agree;quarter-conditioned full-clock CIs also cross0.
Decision:no sign inversion,global residual overlay,fit or new confirmation.
If continuing,next preregister a road-leg/time-aligned innovation and support
audit on original64 only. Associations do not uniquely identify causal shares.
