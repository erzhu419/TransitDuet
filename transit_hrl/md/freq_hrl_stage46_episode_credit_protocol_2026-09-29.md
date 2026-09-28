# Stage 46: full-episode task actor credit

Stage45 enforced training-objective acceptance without a supported native gain.
Stage38 already tested reward source and option/episode cuts. This experiment
instead isolates the actor estimator, leaving critic targets unchanged.

## Frozen Design

Use both task-sham and task-clock Stage42 warmup16 checkpoints (preflight2),
including Adam state; original actors are identical. Per source compare original
normalized option-cut discounted GAE against native undiscounted episode
return-to-go centered by other training episodes' same-time mean. Cross all
option renewals; no learned critic enters the episode actor estimator. Keep
original PPO normalization, clipping, entropy, Adam, learning rate, epochs,
minibatches and gradient clipping. No actor-acceptance gate. Critic targets and
updates retain original task-option GAE. First native batches/task rewards and
first critic weights/Adam must match exactly within each source pair.

Upper/gate actors and values remain fixed, while state-mediated actions may
differ. Lower sampled in training, both sampled/deterministic in evaluation.
Sixteen learning iterations x eight episodes; preflight two x two (LOO needs
two complete paths). Evaluate first/final and shared frozen-before only.
Reuse roots310011/310023/310037/310049/310061/310073/310089/310101; pre310001.
Fresh base11100000+index*10000 (pre11090000), training1..128 (pre1..4),
eval3001..3016 (pre3001..3002). Seeds use SeedSequence(46,root,env,46017)
for policy,46019 plus step for lower noise; shuffle(46,root,iteration).

Eight primary sampled-return endpoints: MC-minus-GAE at first/final, MC-minus-
frozen and GAE-minus-frozen at final, per source. Equal-root paired bootstrap,
65536 draws,seed(46,46046),two-sided Bonferroni8. MC relative benefit alone is
insufficient; MC-minus-frozen must also be positive for learning utility.

## Budget And Limits

Full7680000 native steps/6400 offline trace audits; per root614400 training+
345600 evaluation=960000. Preflight15600/52; zero extra native verification.
Dynamic scheduler node001-node006,9 CPUs/12 GiB per full root; pre2 CPUs/4 GiB.
Only compact JSON local; raw paths/weights remote. No seed extension, selection,
threshold tuning or post-outcome target changes. Reused roots are conditional
development evidence. LOO centering estimates the undiscounted task score before
normalization; clipped multi-epoch PPO has no finite-update improvement guarantee.
