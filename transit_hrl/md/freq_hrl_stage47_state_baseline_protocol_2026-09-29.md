# Stage 47: episode-held-out full-task state baseline

Stage46 did not establish utility for time-LOO episode credit. Isolate its
baseline, using only task-clock warmup16 (pre2): the two prior MC source arms
were identical, and this source exposes causal option age and remaining horizon.
No source/root selection. Compare GAE, time-LOO MC and state-MC, plus shared frozen.
Original PPO settings, GAE critic targets/updates and upper/gate networks remain
fixed. Require identical first native batches/task rewards and critic/Adam updates.

For each query episode, exclude that entire episode from fitting. From other
episodes compute time-mean undiscounted RTG, feature means/stds and RTG residual
std. A fresh ValueNet with original hidden width and zero final head predicts
normalized RTG residual from causal pre-action lower history plus current clocks.
Use original lower Adam LR, epochs, minibatch size, value coefficient and gradient
clip. Add prediction to other-episode time mean. Query labels enter only actor RTG
and post-fit diagnostic MSE. No validation selection, early stopping or carryover.
Baseline initialization uses isolated Torch RNG and local NumPy shuffle, with
SeedSequence(47,root,round,fold,47031). Save predictions/provenance server-only.

Sixteen rounds x eight paths, pre two x four. First/fixed-final and frozen0 only;
both lower deployment modes, sampled primary. Roots310011/310023/310037/310049/
310061/310073/310089/310101; pre310001. Fresh base11200000+index*10000 (pre11190000),
train1..128 (pre1..8), eval3001..3016 (pre3001..3002). Policy seed47/root/env/47017,
lower noise47019 plus step, original shuffle47/root/round.

Seven fixed endpoints: sampled state-minus-MC first/final, state-minus-GAE final,
state-minus-frozen final, MC-minus-frozen final, GAE-minus-frozen final, and first
LOO-minus-state episode-score gradient trace dispersion. Equal-root bootstrap
65536 draws, seed(47,47047), two-sided Bonferroni7. Scores are per-episode means
of logp gradients times original globally normalized advantages; no optimizer
step. The coupled crossfit dispersion is descriptive, not an IID variance bound.
Learning repair requires positive state-minus-MC final AND state-minus-frozen;
a proxy or MSE improvement alone is insufficient.

Full5836800 native steps/4864 offline audits:729600/root=460800 training+268800
evaluation. Pre15600/52. Count auxiliary fitting steps and backward calls apart
from actor/critic updates; zero extra native verification. Dynamic scheduler
node001-node006,9 CPUs/12GiB per full root,pre2 CPUs/4GiB. Only compact JSON local.

## Limitations

Reused roots are conditional development evidence. No seed extension, checkpoint
selection or post-outcome tuning. Crossfit avoids own-label fitting, not finite
sample normalization bias or a multi-epoch clipped-PPO improvement guarantee.
