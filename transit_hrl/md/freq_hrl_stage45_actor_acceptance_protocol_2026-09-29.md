# Stage 45: training-objective actor acceptance

Stage 43 found8/32 first updates lowering the original actor objective. Stage44
found task-sham full displacement worse than a microstep, without establishing
microstep benefit. This intervention tests optimizer fidelity, not a chosen
learning rate or a proof that the PPO objective matches task utility.

## Frozen Design

All four Stage-42 methods; vanilla and accepted-update PPO for each. Start from
fixed warmup checkpoint16 (preflight2), including actor/critic optimizer state.
Same config, original lower PPO,16 learning iterations,8 rollouts per update;
preflight2 iterations/1 rollout. Upper/gate actors and values remain fixed.
For both treatments evaluate original normalized-GAE clipped surrogate plus
original entropy coefficient on the complete current training batch before and
after the original update. Accepted treatment rejects only when this objective
decreases, restoring actor parameters and its entire Adam state. Keep critic
updates. No line search, microstep substitution or evaluation-driven acceptance.
Count executed actor steps and retained steps separately. Rejection is not free.

Reuse all eight original roots310011/310023/310037/310049/310061/310073/310089/
310101; preflight310001. Fresh base11000000+index*10000 (preflight10990000):
training offsets1..128, evaluation3001..3016 (preflight1..2/3001..3002).
Upper/gate deterministic; lower sampled in training, both modes in evaluation.
Lower noise SeedSequence(45,root,env,45019)+step; policy seed uses45017; shuffle
SeedSequence(45,root,learning_iteration). Exact first-batch pairing within each
method. Evaluate first and fixed final checkpoints plus shared frozen-before.

Primary16 sampled-return endpoints: accepted-minus-vanilla at first/final,
accepted-minus-frozen and vanilla-minus-frozen at final, per source method.
Equal-root paired percentile bootstrap,65536 draws,seed(45,45045),Bonferroni16.
Deterministic effects are descriptive. Relative benefit alone does not establish
learning utility; accepted-minus-frozen must also be positive. No exclusions,
checkpoint selection, seed extension, threshold tuning or sequential retesting.

## Budget And Limits

Full15052800 native steps/12544 offline trace audits; per root1228800 training+
652800 evaluation=1881600. Preflight25200 steps/84 audits. Zero extra environment
verification steps. Dynamic scheduler node001-node006,9 CPUs/12 GiB per full
root,8 workers; preflight2 CPUs/4 GiB. Only compact JSON local, raw data/weights
remote. Reused development roots give conditional intervention evidence only.
