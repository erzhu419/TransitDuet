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
