# Stage101: Fixed-UJ Upper Refinement

Stage100's four primary endpoints passed, but its matched-upper specialization did not; at100 both U0-start uppers lost to fixed UJ. Test refinement above that stronger fixed baseline, not another repetition of the U0-start route.

- Both upper means start from the registered final update8 Stage98 joint-call UJ, with the same final Stage99 U0-trained independent/common lowers fixed. No Stage100 learned upper reuse, donor/root/period selection or lower retraining.
- Keep independent upper/lower credit noise, fixed std/values/Adam/source/forecaster/decoder, upper KL0.0005 per update, eight full updates using all64 paths per learner/period/update and fixed-final evaluation. Preflight remains two updates with eight paths.
- Primary endpoints: refined-common minus fixed-UJ/common-lower and refined-independent minus fixed-UJ/independent-lower at both periods. All four corrected CI lower bounds must be positive for the refinement claim.
- Keep all11 actor compositions and 28 contrasts, equal-root bootstrap65,536 / Bonferroni28 with the existing Stage95 bootstrap seed. Matched-upper specialization is a separate four-sign diagnostic: each refined upper must beat the other refined upper on its own same lower. Neither diagnostic substitutes for the primary gate.
- Fresh Stage101 scenario/noise/evaluation roles, disjoint from preflight/full and upstream stages. Preserve all negatives; no reward-based admission, old-cohort pooling or radius/initialization search.
- Preflight: H300, 152 episodes / 45,600 steps, eight upper updates, six checkpoint loads, four UJ initialization checks, zero checkpoint/replay writes.
- Full cohort: H1200, 22,016 episodes / 26,419,200 steps, 256 upper updates, 48 checkpoint loads, 32 UJ initialization checks and 32 final server-only checkpoints. This is additional refinement compute, not equal-total-compute or trajectory-KL evidence.
- Dynamic node001-006: 3 CPU/3 GB preflight, 9 CPU/8 GB full. Pull compact JSON/completion only; full follows native mechanical qualification.

Scope: conditional fixed-lower UJ refinement, not full actor-critic, unseen-task generalization or frequency superiority. Stage100 failures and Stage67 critic-route HOLD remain recorded.
