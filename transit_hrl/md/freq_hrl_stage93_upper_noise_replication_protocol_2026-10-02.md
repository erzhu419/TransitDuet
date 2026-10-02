# Stage93: fresh-sample upper-noise conditioning replication

Stage92 passed all four primary corrected CIs. Replicate on new training/evaluation samples, retraining independent controls rather than reusing Stage90/91 lowers. Four lower means start from L0: independent/shared upper innovations under frozen U0, and independent/shared innovations under frozen final Stage88 UJ. No Stage92 lower weights are loaded.
Each learner retains eight updates,64 episodes/update, periods50/100, horizon1200 and nominal lower KL .00099/.000995. All four use the same new scenario/lower-noise rosters. Shared-noise pairs keep the first rollout unchanged and replay its complete upper innovation tape to the second; independent controls use two unchanged legacy rollouts. Upper/std/values/Adam/decoder/forecaster stay frozen.
Retain all12 evaluation compositions and26 reward contrasts from Stage92, with source_matched/source_fixed and joint_matched/joint_fixed now using the freshly trained controls.32 fresh paired evaluation seeds/root/period, unmodified legacy RNG path. Read only Stage88 UJ/LJ for fixed-upper initialization and secondary comparisons.

## Frozen Analysis
Equal-root bootstrap65536 / Bonferroni26, seed(93,93093). The same four primary shared-minus-independent endpoints must all have positive CI lower bounds. No cross-stage pooling, stronger-period selection, threshold rescue or checkpoint selection; covariance remains descriptive. Same eight teachers, new training/evaluation samples, not eight new teachers.
Full added cost:38,912 episodes /46,694,400 steps;512 lower updates,64 final checkpoints,16 donor loads,64 initialization checks,192 compositions and147,456 extra replay forwards. Per-learner environment/KL budgets match; shared sampling is not equal-total-compute or uniformly variance-reducing.
Native preflight: one root, horizon300, two updates, two scenarios/batch, four evaluation seeds;224 episodes /67,200 steps,16 lower updates,144 replay forwards, no checkpoint writes and no reward gate. Preflight and full seed roles are disjoint.
Scheduler: dynamic node001-006, no pin; full9CPU/8192MiB with8 workers, preflight3CPU/3072MiB with2 workers. Keep checkpoints server-only; pull compact JSON and completion markers.

## Limitations
This is conditional lower learning under fixed uppers, not joint HRL, full actor-critic, teacher-population replication or frequency superiority. Shared innovations can produce different upper actions; Stage91's original repair remains unsupported, Stage67 HOLD and earlier negatives remain unchanged.
