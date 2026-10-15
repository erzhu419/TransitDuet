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

## Outcome

Source `5f3de1bf7a`; tests/smoke `t162425`, shards `t162428`-`t162431`,
aggregate `t162434`: all exit 0. Twelve tests pass; all 12 rollouts reproduce V60
exactly (62,400 decisions). Context-averaged full-return MAE is **0.136658**
versus **0.016456** for zero; all 12 days are worse than zero. One-step target
MAE is 0.027763, which does not establish full-return calibration.
All four cases have 64 actor steps at `lambda=1`; the first positive dual update
to `1.0000743866` is only saved, never used by an actor update. This is missing
adaptive feedback in the smoke, not an absent fixed cost penalty.
Summary: `freqduet/results_freqduet/protocol_v6_v61_cost_calibration/summary/status.json`.

## Limitations / Next Step

Chosen-action labels do not establish the six unexecuted actions' ranking;
absolute bias need not change action preferences. Correlated decisions are not
independent seeds, and the two-day smoke cannot establish long-run dual failure.
Next: freeze the actor, fit signed full returns with complete-day validation,
then test completed-day dual feedback before the next actor block. No multiplier
tuning, relaxed validation, long training or main switch at this stage.
