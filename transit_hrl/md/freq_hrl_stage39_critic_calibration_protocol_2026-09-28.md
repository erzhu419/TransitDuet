# Stage-39 Critic Calibration And Early PPO Drift

Stage-38: all four lower-credit arms harmed final return. Isolate inherited
critic calibration from actor adaptation without changing the PPO recipe.

Five arms: frozen, intrinsic-delayed/calibrated, task-delayed/calibrated.
All use option-terminal credit. Sixteen warmup iterations: calibrated arms
update only lower critic; delayed arms collect the same frozen-policy data but
do not update. Then both learn lower actor/critic for sixteen iterations.
Upper/gate actor/critic stay frozen; same inherited networks, optimizer recipe,
reward scales and learning rates. All levels sample on-policy in training.
Eight episodes x1200 steps each iteration; equal actor-update and environment
budgets, with additional critic optimization charged in calibrated arms.

Keep every checkpoint at0/16/17/20/32; no selection or early stopping.
Evaluate sixteen new paired paths at each checkpoint, both deterministic
and lower-only sampled; upper/gate deterministic in both deployment modes.
One disjoint initial stochastic probe per cell supplies fixed states and
discounted Monte Carlo targets with actual option cuts; never used to train.
Record critic MSE, Gaussian KL, squashed mean-action drift and per-update GAE.

Roots310011/310023/310037/310049/310061/310073/310089/310101 are reused.
Base10400000 + root-index x10000: train+1..256, probe+2001, eval+3001..3016.
Shuffle SeedSequence(39,root,iteration); paired lower sampling uses
SeedSequence(39,root,environment-seed,39019). Eight adjusted return endpoints:
four final arms vs frozen, two final calibration effects, two first-update
calibration effects. 65536 root-paired draws, seed(39,39039), Bonferroni8.
Sampled-mode curves and probe MSE are descriptive diagnostics, not new gates.

Full40 cells:20064000 method steps plus576000 verification steps. Preflight
root310001/base10390000: warmup2/learn2, one rollout, two evaluation paths,
snapshots0/2/3/4; five cells/33000 method steps plus15000 verification steps.
Dynamic scheduler node001-node006, full cpu9/ram12GiB; raw data/weights remote.

## Limitations

Conditional development, not independent confirmation or algorithm victory.
Monte Carlo error is on one noisy fixed-policy trajectory, not exact value
truth; post-update off-policy probe MSE is diagnostic. Task reward also removes
the intrinsic action penalty. Inadequate warmup does not refute calibration;
it also does not authorize outcome-driven warmup, seed or budget extensions.
