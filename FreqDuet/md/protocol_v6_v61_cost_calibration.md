# V61: remaining-CV cost calibration and dual timing

Freeze the V59 post-64 actor, current/target cost critics, upper actor and B200
deployment state. Repeat only V60's 12 updated/stochastic full-day rollouts with
identical scenario and private sampler seeds. No optimizer steps or main switch.

## Diagnostic

Reuse V60 control through a read-only decision audit. Capture exact completed
arrival moments before every chosen action, current selected-action cost Q,
policy-averaged current cost value and policy-averaged target cost value.
Only after the day ends, label remaining cost as `CV_day^2 - CV_prefix^2`;
retain negative returns and same-clock continuation. Compare full-return
calibration with the one-step target and a zero-value baseline, by rollout,
day thirds and the final decision. Exact V60 metrics, action counts and
physical stream alignment must hold. Large per-decision CSV stays on server.

Audit V59's recorded optimizer steps, learning rates, lambda values used by
actor updates, and whether any update used the first positive-violation dual.
Average the three sampler repeats within each of four development contexts.
Run tests and one exact replay through scheduler before the full server batch.

## Limitations / Next Step

These are chosen-action value labels under the current stochastic policy, not
counterfactual rankings of the six unexecuted actions. Decisions are correlated;
no seed-level inference is made from their count. The actor has only 64 updates,
and a delayed dual in this two-day smoke does not by itself prove a long-run
algorithm defect. Use calibration and timing evidence to select critic warm-up
or update-order repair before multiplier changes and long training.
