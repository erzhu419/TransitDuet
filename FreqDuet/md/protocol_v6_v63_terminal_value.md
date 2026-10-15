# V63: exact prefix, learned terminal CV value

Represent remaining regularity cost as
`Q_cost(s,a) = U(s,a) - CV_prefix^2`, where `U` predicts terminal `CV_day^2`.
Subtract the exact completed-arrival prefix, not an inferred or future value.
This state-only subtraction cannot itself change action preferences.

Reuse only V62's four fitting rollouts per frozen V59 actor. Keep the original
network, initialization, learning rate, 2,048 steps, batch 256 and V62 batch-draw
seed. Change only targets to terminal CV squared and the analytic subtraction.
Save terminal weights under a distinct semantic key, never as a direct cost Q.
Fit all four models before evaluation; no validation-driven stopping or tuning.

Collect 16 fresh frozen-policy rollouts: four actors x demand seeds
45207/45307/934211/934311, sampler repeat 2. Compare on the same states/actions
against V62's fixed direct-return model, V59 frozen critic, zero prediction and
`mean(training final CV^2) - current CV^2`. Gate: both MAE and RMSE improve over
every baseline on every validation day, for all four actors. No actor or dual
updates, relaxed gates or main switch. All compute through server scheduler;
state arrays and checkpoints remain there.

## Limitations / Next Step

The four demand days are shared across actors (two original training seeds).
Chosen-action calibration is not unexecuted-action evidence or control benefit.
Record seven-action prediction spread without treating it as counterfactual
truth. Only after calibration passes, test matched action differences before
changing the control policy. Preserve V62's negative result.
