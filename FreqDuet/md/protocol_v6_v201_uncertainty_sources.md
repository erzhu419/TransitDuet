# V201: original64 uncertainty localization

Read existing moment statistics only;four primary mode/scope cases from V200.
Freeze V199 masks,V191 weights,original bins,19-day threshold,parent/day centers
and nominal denominators. Reproduce V200 points/intervals on the same10000
shared-actor/independent-half-day draws,seed20260803,before interpretation.

For actor conditional readout theta_i(D),bootstrap product variance partitions:
actor=Var_i(E_D theta_i)/12;day=Var_D(mean_i theta_i);
interaction=E_D Var_i(theta_i-E_D theta_i)/12.
Their sum integrates multinomial actor resampling exactly conditional on the
10000 day draws. Compare,but do not force equality,to finite crossed-sample
variance. This is bootstrap uncertainty,not causal/physical variance.

Actor and exact(direction,road,reference-hour) key allocations are signed
Cov(contribution,total) in the same crossed draws. Preserve negative offsets;
sum allocations must equal total variance. Key grouping occurs AFTER frozen
actor-level decomposition,never pools support. These attribution views must
not be added to the actor/day partition. Keep all allocations server-side.

Shared-day deletion within each half uses1/31 day weights and unchanged masks
and V191 factors,other half fixed. It is sensitivity only,not selected-mass
renormalization,a new gate,or grounds to remove days. Report all64 days server-
side;summary top3 by absolute shift/allocation is descriptive,not confirmation.
Low-support contribution remains separate and aggregate. No additional tests
of selected actors,keys or dates;no V200 gate reopening.

Five server tests target per-key additivity,orthogonal variance channels,
signed cancellation,day deletion and undefined resamples. High/manual scheduler
on node004;local summary only. No raw scan,interval/predictor fit,confirmation,
replay,training or change to V151/V146/V172,3600s,.1/.1 tails or main/actor/dual.
