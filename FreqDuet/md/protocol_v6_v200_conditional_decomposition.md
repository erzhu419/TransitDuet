# V200: fixed-support conditional cross-moment decomposition

Original64 days2951001-2951064 only. Read V199 moments,its sealed19-day-in-EACH-
half masks,and saved V191 parent-key balance. No raw archive scan. Center J and
first-road Y using FULL V194 base-key/day means before excluding any cell.
Pool centered moments within each half/actor/key/bin. For supported cells:
M=original balanced mass,t=E[(J-mu_parent_day_J)*(Y-mu_parent_day_Y)|cell].
Delta(sum M*t)=sum DeltaM*average(t)+sum average(M)*Delta(t).
Split t into conditional covariance and product of conditional centered means;
report both subcomponents without a new projection or gate.

Keep low-support mass and centered-cross contribution separate,without cell
decomposition or selected-mass renormalization. Supported+unsupported must
reproduce full parent contribution. Preserve all fixed bins,masks and factors.
Use10000 shared-actor/independent-half-day draws,seed20260803;recompute pooled
cell means per draw. Eight two-sided primary mix/conditional contrasts across
two modes and clearance/hour13_moving_launch use alpha=.05/8;other intervals95%.
Hour13 overlaps clearance. Undefined bootstrap cell means remain unknown;
no mask replacement or discarded draws. Existing V195 gates remain unchanged.

Tests target full-parent/day centering,pure mix vs conditional change,conditional
covariance/mean-product algebra,no renormalization and missing bootstrap cells.
All calculations/tests run via high/manual scheduler,node004;summary only local.
Frozen V151,V146,V172,3600s,.1/.1 tails and main/actors/critic/dual unchanged.
No ETA interval/predictor fit,new confirmation,replay or training.

Limitations:reused calibration days and partial supported mass. Conditional
cross-moment change can still reflect within-bin state/day mixture,not a
conditional-law or causal control effect. No extrapolation to the whole cohort.

## Result and decision

t210909/source52e8a70164 completed high/manual-dispatch on node004,exit0.
Five tests passed;all6 cases reproduced V199 support/exposure and parent-day
identities. All10000 draws valid;only65972-byte summary fetched:
freqduet/results_freqduet/protocol_v6_v200_conditional_decomposition/readout_52e8a70164/summary/status.json.
Original nominal-mass units s^2;family intervals below are99.375%:

| Mode/scope | Mix [family CI] | Conditional cross-moment [family CI] |
| --- | --- | --- |
| old/clearance | -0.10 [-1.62,1.64] | 3.27 [-9.17,15.76] |
| handoff/clearance | -0.29 [-2.04,1.42] | 4.16 [-8.48,16.81] |
| old/hour13_moving_launch | 0.01 [-1.19,1.18] | 3.91 [-5.80,14.10] |
| handoff/hour13_moving_launch | -0.06 [-1.28,1.07] | 4.04 [-5.39,13.98] |

All8 primary intervals cross zero. Conditional covariance/mean-product changes
also have diagnostic95% intervals crossing zero. Clearance unsupported delta:
old6.03[.89,11.70],handoff5.76[.86,11.12],diagnostic95%,not primary confirmation.
The supported subset retains only26-38% of balanced mass;these point estimates
cannot explain the full cohort or prove conditional-law change. No correction
is justified. Next useful diagnostic:existing actor/key contribution and
day-level uncertainty localization,not new bins,thresholds or calibration fits.
