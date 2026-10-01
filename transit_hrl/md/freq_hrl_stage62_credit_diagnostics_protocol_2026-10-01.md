# Stage62 Credit/Critic Diagnosis

Stage61 passed mechanics but all12 native return intervals crossed zero. Diagnose the saved post-first-update backtracking critics before changing credit or critic training. Retain all eight roots, periods50/100 and zero_train/joint_ppo. Use Stage57's first eight training paths and all16 fresh Stage61 deterministic paths per arm/period; preflight root310001 uses2+2. No path selection, training, policy action inference or new environment sampling.

Reconstruct the same causal upper state and lower value context directly from existing raw archives. Compare lower GAE under option-terminal, option trace-cut with continuing bootstrap, and episode-continuing boundaries using the unchanged SMDP core. Measure normalized advantage correlation/sign disagreement and residuals at artificial boundaries. Evaluate saved lower critics against discounted option/episode Monte Carlo returns and option GAE targets; upper against charged discounted episode Monte Carlo and GAE targets. Pool samples within root/split; report equal-root descriptive means and all roots, with no performance gate.

Full incremental cost:768 archived episodes,32 checkpoint loads,921600 lower and13824 upper feature/value rows;2304 lower/768 upper batched value passes,3072 GAE and2304 MC calls. New native/actor/optimizer/fit/forecaster/checkpoint-write counts0. Reuse server archives; do not reconstruct the full Stage57 optimizer path again. Scheduler only, dynamic node001-node006,2CPU/4GB per task. Tests check exact native feature/return reconstruction, distinct bootstrap semantics, state immutability, rosters and accounting. Source/protocol and preregistration are committed before full outcomes; only compact JSON is local.

Next intervention is chosen after these diagnostics, tested separately and keeps KL backtracking unchanged. No seed expansion or KL sweep is justified by Stage61.

## Limitations

Advantages use saved post-update values, not the original first-update sampling values. Training paths fitted the shared critic but their actors were pre-update; held-out paths execute a deterministic post-update policy, not the stochastic training policy. Continuing-bootstrap calculations reuse an option-trained critic and are sensitivity diagnostics, not unbiased episode-value estimates or a causal credit ablation. Realized MC errors are noisy policy/distribution-dependent diagnostics, not proof of a reward improvement or a defective option objective.

## Execution

Implementation/protocol committed atcc150f5594. Task`t116561` passed all4 focused tests on node006 in7.642s, exit0: exact native feature/macro-return reconstruction, three boundary semantics, frozen state/no actor/environment/update calls, seed roster and cost accounting. No existing algorithm modules changed. Real archive preflight is next; no diagnostic conclusion yet.
