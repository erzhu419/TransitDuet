# Stage69: independent frozen-policy credit confirmation

Stage67 improved critic fit but failed its credit gate; Stage68 showed poor cross-half gradient repeatability. Stage69 separates sampling noise from GAE value-error propagation. The Stage67 native HOLD remains unchanged.

- Freeze the same eight roots, periods 50/100, zero_train/joint_ppo execution, Stage55 actors/forecaster and Stage64/67 critics. Reproduce the original first-training value/GAE probe.
- Sample four independent batches of eight episodes per case, using the original stochastic training distribution without updates. New reset/action seeds are disjoint from historical fit/probe/evaluation roles and paired across cases. Preflight uses two batches of two episodes.
- Use one time-only baseline for both critics: discounted remaining mass times the Stage67 first-calibration mean reward rate. Reference gradients are raw, uncentered MC-minus-common-baseline gradients. Compare independently normalized PPO GAE directions, with entropy reported separately, against disjoint MC batches and compare the old probe against new MC batches.
- Report raw independent episode-gradient covariance, unbiased squared-signal estimate and debiased mean SNR without truncating negative estimates. Batch-pair comparisons are dependent descriptive observations, not extra roots or CI samples.
- Attribute TD residual and GAE-minus-MC advantage to the exact filtered future value-error identity, masking true episode terminals. Report global, renewal and tail diagnostics. No lambda, learning-rate, seed or gate search.
- Full budget: 1,024 fresh episodes / 1,228,800 primitive steps; 256 reconstructed anchor episodes. No actor/value optimizer steps, critic fits or checkpoint writes. Scheduler uses node001-node006 dynamically, 9 CPU / 6 GiB per full root, 3 CPU / 3 GiB preflight. Pull completion markers and compact JSON only.

## Limitations
This is independent sampling under reused teacher-initialized development policies, not a changed-policy reward trial, a true-gradient oracle, OOD evidence or frequency-superiority proof. The raw MC reference uses the existing uniform-time discounted-return PPO convention, not the gradient of undiscounted native episode reward. Lower gradients condition on sampled frozen upper plans. Diagnostics cannot release a failed adoption gate.
