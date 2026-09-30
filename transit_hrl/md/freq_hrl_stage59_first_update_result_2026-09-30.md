# Stage59 First-Update Result

All eight full tasks `t109682`-`t109689`, qualification `t109964` and metadata inspection `t109974` completed with exit0. **Decision: HOLD; no native performance trial.** The frozen KL guard prevented displacement by freezing every lower actor, not by producing a smaller learning update.

Plain first-update replay matched Stage58 exactly. Paired final critics and critic Adam states matched exactly; rejected steps restored actor and Adam exactly. The frozen budget and cost accounting passed.

Equal-root means across eight roots, for the first update only:

| Period | Arm | Plain lower KL | Guarded lower KL | Retained / attempted lower steps |
| --- | --- | ---: | ---: | ---: |
| 50 | zero_train | 6.469716 | 0 | 0 / 320 |
| 50 | joint_ppo | 5.739111 | 0 | 0 / 320 |
| 100 | zero_train | 0.028248 | 0 | 0 / 320 |
| 100 | joint_ppo | 0.029707 | 0 | 0 / 320 |

All32 lower cases had zero retained steps and zero action-mean movement:1280/1280 attempts rejected. All64 upper steps were retained; final upper conditional mean KL ranged0.00912-0.01763 and matched plain PPO. The preregistered nonzero-actor prerequisite therefore failed.

Posthoc inspection read16 source checkpoints on the server only. All32 initial actor optimizers had actual LR3e-4 matching configuration and empty PPO Adam state. Across rejected lower attempts, candidate mean KL ranged0.117328-2.574241, all above0.02. This is not an LR bookkeeping error or inherited-Adam problem; a fixed-size proposal followed only by rejection cannot produce lower learning here.

Frozen reconstruction cost:4352 episodes,5222400 lower/78336 upper archived calls,1024 critic warmup updates,96 observed updates; diagnostic distribution/value passes192/192 and GAE calls96. Executed lower actor/value steps2560/23040 and upper actor/value steps128/2176. Guard cost:1392 distribution passes,2688 state snapshots,2560 rollback checks. Source loads16 clones/eight forecasters; the additional16 posthoc checkpoint reads are separate. New native/evaluation/fit counts remain0.

Next intervention: freeze an actor-only conditional-KL backtracking comparison at the same0.02 budget, reducing rejected proposals while restoring actor and Adam between retries and preserving the original critic updates. Report retry cost and actual movement. Do not widen the budget, reset Adam, change credit, select roots/periods or launch native performance on this HOLD result.

## Limitations

Fixed-batch conditional KL is not a trajectory or population bound. This archive-only result is not reward-improvement evidence and leaves Stage57 performance gates unchanged. Upper critic repair remains separate.

Compact data: [compact_summary.json](../results/pointmaze_first_update_stage59_full_20260930_r1/compact_summary.json).

