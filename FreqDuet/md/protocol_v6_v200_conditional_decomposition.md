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
