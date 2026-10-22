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

## Result and next decision

t210914/source5d73c20fd9 completed high/manual-dispatch on node004,exit0.
Five tests passed;all4 cases reproduced V200 points/intervals(max error7.11e-15)
and variance/allocation identities. Only53271-byte summary fetched:
freqduet/results_freqduet/protocol_v6_v201_uncertainty_sources/audit_5d73c20fd9/summary/status.json.

Shared-day variance accounts for95.98-97.09% of mix uncertainty and98.49-98.85%
of conditional-term uncertainty. Conditional actor main effect0.54-0.78%,
interaction0.60-0.73%. This is not predominantly an actor-heterogeneity problem.
The exact keys(0,7,13),(0,12,13) jointly allocate80.77-88.17% of conditional
crossed variance by SIGNED covariance allocation,not independent key variances.
Conditional product-measure/crossed variance difference is below0.18%.

Clearance conditional sensitivity is largest on2951044,2951038,2951013;
moving-launch on2951032,2951033. Maximum single-day deletion changes the point
by1.46-1.69s^2 and does not reverse its positive sign in any case. These are
descriptive dates,not exclusions. SD/point units s^2;variance units s^4.

Decision:V200 remains uncertain;do not fit a correction or add actor training
to address this shared-day uncertainty. Next inspect existing day-level J and
first-road error co-moments inside the two dominant fixed keys,including
supported/unsupported contributions and cancellation. No new gate or fitting.
