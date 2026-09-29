# Stage 48: full-task actor episode-KL budget

Stage47 improves prediction descriptively, not native utility. Isolate update
size: task-clock Stage42 warmup16 (pre2), original GAE, time-LOO full-task MC,
and the same MC with training-path episode KL backoff; shared frozen reference.
Retain original PPO epochs/minibatches/clipping/entropy/gradient clipping.

For each learning round, try original actor LR then successive halves up to
12 backtracks. Restore lower actor/critic and both Adam states before each
trial and use identical shuffle. Compute exact old-to-new Gaussian KL at
pre-action histories in float64, sum over each complete episode; accept first
trial whose maximum episode sum is at most **0.1 nats**. This fixed complete-path
budget tests trajectory-scale displacement rather than a per-step mean.
No objective floor or evaluation access. Retain the accepted actual Adam path,
restore original LR for next round; if all13 fail, restore actor/Adam. Always
retain exactly the first original GAE critic/Adam update. Charge all executed
actor/critic trial steps separately from retained steps and KL check calls.
First native batches, rewards and critic/Adam must match across all three arms;
the first full MC candidate must also match exactly before backoff.

Sixteen rounds x eight paths; pre two x four. First/fixed-final and frozen0;
sampled lower primary, deterministic descriptive. Roots310011/310023/310037/
310049/310061/310073/310089/310101; pre310001. Fresh base11300000+index*10000
(pre11290000),train1..128 (pre1..8),eval3001..3016 (pre3001..3002).
SeedSequence48/root/env/48017 policy,48019 lower noise plus step; shuffle48/root/round.
Upper/gate parameters fixed; state-mediated actions can change.

Six primary sampled-return endpoints: bounded-minus-MC first/final,
bounded-minus-GAE final, bounded-minus-frozen final, MC-minus-frozen final,
GAE-minus-frozen final. Equal-root paired bootstrap65536 draws,seed(48,48048),
two-sided Bonferroni6. Repair requires positive bounded-minus-MC final AND
bounded-minus-frozen. KL feasibility and training objective gains alone do not.

Full5836800 native steps/4864 offline audits;729600/root (460800 train+268800 eval).
Pre15600/52. Original retained full budget at most15360 actor/15360 critic steps;
up to76800 executed steps each with13 KL trials, not a compute-saving method.
Zero extra native verification. Dynamic scheduler node001-node006,9CPU/12GiB
per full root,pre2CPU/4GiB; raw/weights server-only,compact JSON local.

## Limitations

This is an empirical on-training-history constraint, not a population trajectory
KL bound or reward guarantee. Reused roots are conditional development evidence.
No budget tuning, seed extension, checkpoint selection or candidate selection
by fresh evaluation. Failed trials are charged; rollback is not free training.
