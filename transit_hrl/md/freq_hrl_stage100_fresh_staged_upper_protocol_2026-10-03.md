# Stage100: Fresh Staged Upper

Repeat the Stage94/95 staged-upper rule with the eight Stage96 teachers, fixed Stage97 decoders and rebuilt Stage99 lowers. Stage99's all-four conditioning claim is supported; this separate experiment tests the complete lower-then-upper route on the same new cohort.

- Fixed donors: both final update8 U0-trained Stage99 independent/common lower means and the final Stage98 joint policy for diagnostics. Keep every root; no donor or period selection.
- Both upper means start from U0. Train with original independent upper/lower noise, with the corresponding learned lower bit-exact frozen. No shared upper replay, std/value/Adam/forecaster updates or decoder recalibration.
- Unchanged upper conditional KL 0.0005 per update, eight full updates and all64 native credit paths per learner/update/period; preflight two updates with eight paths. Fixed-final evaluation only.
- Keep all11 actor compositions and all28 Stage95 reward contrasts, including U0, fixed UJ, independent-trained and crossed-actor controls. Same 65,536-draw equal-root Bonferroni family and bootstrap seed; all four primary CI lower bounds must be positive for the global staged claim.
- Stage100 scenario/noise/evaluation namespaces are fresh and disjoint between preflight/full and upstream stages. No old-cohort pooling or reward-based preflight admission.
- Preflight: H300, 152 episodes / 45,600 steps, eight upper-mean updates, six checkpoint loads and no checkpoint writes/replay forwards.
- Full cohort: H1200, 22,016 episodes / 26,419,200 steps, 256 upper-mean updates, 48 checkpoint loads and 32 final server-only checkpoints. Prior teacher/lower/joint preparation costs remain separately reported; this is not equal-total-compute or trajectory-KL evidence.
- Scheduler dynamically chooses node001-006: 3 CPU/3 GB preflight, 9 CPU/8 GB full. Pull compact JSON/completion only; submit full only after native mechanical qualification.

Scope: teacher-initialized staged MC mean learning on the same native task, not full actor-critic, unseen-task generalization or frequency-superiority proof. Stage67 critic-route HOLD is unchanged.
