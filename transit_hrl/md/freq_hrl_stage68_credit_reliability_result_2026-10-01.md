# Stage68 Result and Next Step

All eight roots completed in29.5-32.2s each; complete-roster qualification passed. All64 Stage67 value/GAE probes reproduce, full credit-gradient metrics match within the frozen numerical tolerance, and all model/Adam states stay bitexact.5 analytic/regression tests passed. No native sampling, fitting, optimizer steps or checkpoint writes. [Full compact statistics](../results/pointmaze_credit_reliability_stage68_full_20261001_r1/compact_summary.json).

Equal-root mean cosines across all35 disjoint4/4 partitions (ordinary -> factored; partitions are dependent, not independent CI samples):

| Period / Arm | GAE Mean-Gradient Repeatability | MC Mean-Gradient Repeatability | Cross-Half GAE/MC |
| --- | ---: | ---: | ---: |
| 50 / zero_train | -.008 -> .046 | -.022 -> .031 | .046 -> .124 |
| 50 / joint_ppo | .064 -> .065 | -.064 -> .168 | .036 -> .068 |
| 100 / zero_train | .002 -> -.094 | .010 -> .124 | .086 -> .044 |
| 100 / joint_ppo | .085 -> -.040 | .127 -> .210 | .159 -> .050 |

Same-batch GAE/MC mean cosines of.54-.74 shrink to.04-.12 on disjoint halves for the factored critic. Its MC repeatability improves in all four groups, but GAE deteriorates at period100. Changing the fitted MC baseline also rotates its full mean-gradient reference (between-critic MC cosine.40-.64). Neither the same-batch MC comparison nor better value fit certifies a useful policy update. This does not establish that MC noise alone caused the failure.

Next freeze a genuinely independent policy-gradient confirmation: new trajectories from the unchanged actor, a common causal baseline and explicit gradient noise/TD-residual diagnostics. Retain all current roots and results; no lambda/LR/seed/gate sweep or actor deployment. Independent confirmation is not implemented or launched yet. Stage67 native HOLD stays unchanged. Cost:256 archives,307,200 lower reconstructions,1024 score forwards/4096 backwards,1120 dependent partitions. Only30.5KB preflight and245.4KB full JSON pulled; traces/weights stay on servers.
