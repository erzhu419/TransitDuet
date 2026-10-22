# V202: within/between-day covariance at two frozen keys

Original64 calibration days only;keys(0,7,13),(0,12,13),four V200 primary cases.
Read V199/V194 moments,V191 fixed weights/masks and V201 key references only.
No raw trajectory access. Preserve bins,19-day-in-EACH-half admission,original
denominators,all12 actors and full-parent/day centering.

Raw parent covariance=mean daily within-parent covariance+between-day parent-
mean covariance. V200 uses the former,not their sum. Within each supported
actor/key/bin after full-parent/day centering,half cross mass=W+B+P:
W=mean daily within-cell covariance;B=between-day centered cell-mean covariance;
P=pooled centered cell-mean product. V200 conditional covariance=W+B. Apply its
unchanged symmetric mass/conditional identity to all3 parts. Keep unsupported
mass/centered cross contribution aggregate;no conditional readmission or pooling.

Use saved I,A,J=I-A to retain exact signed covariance cancellation for every
point/draw. Same10000 shared-actor/independent-half-day draws,seed20260803;
reproduce V201 fixed-key points and own bootstrap variances. New95% intervals
are exploratory diagnostics on selected development keys,not a new gate.
Four server tests cover centering,within/between distinction,cancellation and
V201 additivity. High/manual scheduler,node004;small summary only local.

Limitations:temporal/covariance algebra does not identify causal mechanisms or
predictor update bugs. Raw parent between-day covariance is different from
centered cell-mean covariance. No extrapolation beyond fixed supported mass.
No interval/predictor fit,new confirmation,trajectory scan,replay or training;
V151,V146,V172,3600s,.1/.1 tails and main/actor/dual stay frozen.

## Result and next decision

t210923/source8829065fe1 and saved-draw-only supplement t210924/source9d1e2d7f9c
completed high/manual-dispatch on node004,exit0;4+2 tests passed. V201 fixed-key
points/variances reproduced(max point error1.33e-15). Fetched82844+14534 bytes:
freqduet/results_freqduet/protocol_v6_v202_key_day_covariance/{audit_8829065fe1,cancellation_9d1e2d7f9c}/summary/status.json.

Key12 conditional J change2.33-2.61s^2 splits into daily within.76-.80,
centered between-day1.17-1.44,and pooled mean product.37-.43. Key7 conditional
change is-.93 to-.72 in clearance,but+.92 to+.98 in moving-launch;its centered
between-day term changes direction across these scopes. Raw full-parent within-
day covariance estimates stay negative in both halves for both keys;positive between-
day parent-mean covariance can reverse the pooled point sign. This is NOT a
sign reversal of the within-day relationship or a term to add back into V200.

Matched-draw signed variance allocation:Key7 within49.79-54.10%,between38.52-
42.32%,mean product7.39-8.08%;Key12 between51.43-54.01%,within28.37-30.12%.
I contributes106.75-111.21%,minus_A contributes-11.21 to-6.75%,preserving their
negative covariance. These are dependent allocations,not independent variances;
V201 shared-day bootstrap uncertainty is not identical to between-day covariance.

J within/between/mean-product change95% intervals all cross zero. No correction
or control promotion is justified. Next:use saved daily raw/centered moments
to separate parent/day-centering terms from cell-mean co-movement inside the
same supported bins;do not change bins,thresholds or fit a new interval.
