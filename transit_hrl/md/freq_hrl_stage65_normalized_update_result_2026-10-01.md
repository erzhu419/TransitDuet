# Stage65 Normalized Update Status

Four new tests t116814 passed in62.141s; four Stage64 regression tests passed in t116810. Initial fixture comparison/snapshot failures remain in the test records. Actual native preflight t116815 and qualification t116816/node006 exited0 and passed the mechanical gate.

All eight preflight actor updates are nonzero, with32/32 Adam steps retained. Conditional mean KL0.008210-0.008726 stays below0.02; action-change RMS0.044995-0.046389. Stage64 critic probes reproduce exactly, normalization/Adam units remain consistent, and upper networks/Adam are common and frozen. No roots or cases were filtered.

New preflight work:8 archived episodes,2400 lower/36 upper reconstructions plus2400 extra critic calls;32 actor/32 value optimizer steps (16 MC-supervised),40 guard distribution passes;24 native episodes/7200 primitive steps,108 upper calls and84 plan OLS/ridge plus matching audits. No new training paths or forecaster fits. Upstream Stage63/64 work is recorded separately. Only20.2KB compact JSON pulled; all traces/checkpoints stay remote.

Full preregistration d5ab6f0fc2 is retained. Eight roots t116817-t116824 ran on node001-node006 without pins; qualification t116843/node004 completed with exit0. The mechanical gate passed, no actors froze,2558/2560 proposed Adam steps were retained and2 rejected. Native evaluation completed1536 episodes/1843200 primitive steps. All12 equal-root Bonferroni-corrected reward intervals cross zero: repair_gain_gate and training_gain_gate failed.

| Period / Arm | Candidate Minus Raw GAE: Mean [CI] | Candidate Minus Frozen Lower: Mean [CI] | Positive Roots vs Frozen Lower |
| --- | --- | --- | --- |
| 50 / zero_train | -0.27 [-1.87,1.60] | -0.60 [-1.91,0.32] | 3/8 |
| 50 / joint_ppo | 0.04 [-1.61,1.81] | -0.96 [-2.27,0.50] | 2/8 |
| 100 / zero_train | -1.09 [-4.98,1.92] | -1.75 [-4.75,0.14] | 1/8 |
| 100 / joint_ppo | -0.95 [-3.32,1.50] | -1.70 [-4.32,1.04] | 3/8 |

Candidate critic EV remains0.491-0.736 before the actor update and0.716-0.872 afterward; this fit improvement did not establish a native reward gain. Period50/joint candidate-minus-clone is+3.52 [-0.56,7.81], while candidate-minus-frozen-lower is negative: shared upper gains must not be attributed to lower learning. Next diagnose the actual pre-update credit/actor gradients, not expand seeds or select a favorable period.

Completion-only synchronization and complete-roster qualification are now connected; t116829/node005 passed both dependency and archive-to-native tests (73.032s). Eight markers total792 bytes; full compact statistics126.8KB. No traces/checkpoints pulled. [Compact results](../results/pointmaze_normalized_update_stage65_full_20261001_r1/compact_summary.json).

## Limitations

Full first-update native reward improvement is inconclusive. Full-training stability, OOD and frequency-specific superiority remain unconfirmed. These reused development roots and deterministic deployment are not a generalization test.
