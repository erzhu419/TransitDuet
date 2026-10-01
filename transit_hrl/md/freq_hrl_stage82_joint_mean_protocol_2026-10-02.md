# Stage82 Joint Mean Protocol

Question: do the Stage81 upper/lower mean-credit gains survive simultaneous
closed-loop changes under the same total conditional KL budget?
Freeze Stage78 decoder, sources/Adam, both std vectors, critics and forecaster.
Reuse scenario MC credit on fresh namespace82 seeds; no reward-based selection.

Total budget.001 is the sum of per-level source-state average Gaussian KL,
not trajectory KL. Single-full uses.001 on one level; joint uses.0005 each.
Single-half controls use the identical actors composed into joint candidates.
Register plus/minus and both cross-sign combinations, source and zero residual:
14 variants,50/100 periods,8 roots,32 independent evaluation seeds per root/period.
Credit: two disjoint batches of16 scenarios with2 independent action-noise replicates.
All60 reward/interaction endpoints share equal-root bootstrap65536/Bonferroni60.
Interaction = joint++ + source - upper-half+ - lower-half+, paired by evaluation seed.
No radius/alpha sweeps, actor adoption, optimizer steps or checkpoint/trace writes.

Before full runs, preflight checks exact joint composition/std freeze, objective,
common-noise pairing, fixed radii and budget; its rewards do not select settings.
Full budget:8192 episodes /9,830,400 steps, including1024 credit episodes;
3072 score forwards,9216 backwards,2448 Fisher JVPs,4896 exact-KL forwards.
Scheduler dynamically places9CPU/8GB tasks on node001-006 (preflight3CPU/3GB).
Only completion markers and compact JSON return locally.

Next decision: compare joint/source, both half-budget conditional improvements,
equal-total-budget singles, interaction and zero-residual controls before training.
This single-step experiment does not lift Stage67 HOLD or establish frequency superiority.
