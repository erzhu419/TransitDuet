# Stage-37 Level Updates And Gate Deployment

Stage-36 gate-only and upper/lower-only updates both harmed final reward.
Freeze two complementary diagnoses before changing objectives or architecture.

Training: all-frozen, upper-only, lower-only, upper+lower, trained fixed50.
Gate stays at the same Stage-35 initialization in all gated arms. Each enabled
level updates both actor and critic with the existing SMDP PPO; disabled
weights and optimizers remain unchanged. Networks, rewards, cost1, cadence,
gamma0.995, lambda0.95, learning rate3e-4 and four epochs remain unchanged.
128 iterations x8 paths x1200 steps; selection0/32/64/96/128 on8 paths;
evaluate32 paired paths for final128 and selected weights. Final is primary.
NumPy shuffle SeedSequence(37,root,iteration); initial weights unchanged.

Deployment diagnosis: use the Stage-36 final checkpoints for frozen, gate-only
and joint, with no retraining or checkpoint selection. Upper/lower actions
remain deterministic. Compare gate probability threshold0.5 with Bernoulli
sampling at the original probabilities. Each of32 new paths has one threshold
trajectory and four sampled trajectories per cached model. Random streams
are paired across models: uint32 SeedSequence(37,root,path,stream), then Torch
seed(base+observed-step) per eligible decision. Record states, probabilities
and actual actions. Lower runs every step; no previews; max age100/check25.

Eight inherited roots310011/310023/310037/310049/310061/310073/310089/310101.
New environment base8400000 + root-index x10000: training+1..1024,
selection+2001..2008, main evaluation+3001..3032, gate evaluation+4001..4032.
Forty training cells plus eight cached-gate cells: 58,800,000 method steps;
153,600 verification steps separately. Cached evaluation has zero updates.

Nine primary reward contrasts: upper-only, lower-only, both versus frozen;
upper/lower interaction; sampled-minus-threshold for each cached model;
gate-only-minus-frozen and joint-minus-frozen under sampled gates. Equal-weight
root means after path/stream averaging, 65536 bootstrap draws, seed(37,37039).
Two-sided percentile Bonferroni intervals across all nine endpoints, including
both phases. Sampling improvement alone is not a learning improvement.
ISE, calls, utility, fixed50 and selected checkpoints are secondary context.

Separate preflight310001: five training arms, two iterations, two paths per
cohort; cached diagnosis two paths/two streams per model. Method24900 steps,
verification4800. Readiness tests execution, component freezes, actual gate
sampling and native replays, never performance. Full/preflight dispatch only
through scheduler's dynamic node001-node006 pool, cpu9/ram12GiB full,
cpu2/ram4GiB preflight. Stage source directories only; arrays/checkpoints
remain remote and only compact JSON returns locally.

## Limitations

Conditional development with reused controller roots/cached checkpoints,
not independent confirmation or algorithm superiority. No root deletion,
budget extension, threshold tuning or outcome-driven mode selection. Earlier
failed results stay failed; both deployment modes are reported.
