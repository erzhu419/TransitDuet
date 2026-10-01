# Stage67 Result and Next Step

All eight roots and t116915/node006 completed with exit0. Same-budget critics reproduce all32 Stage64 controls exactly; actors/upper and their Adam remain frozen. Full fit gate passes32/32. Credit gate fails; native_trial_prerequisite stays HOLD.

Equal-root means (ordinary MC -> factored MC):

| Period / Arm | Global MSE | Tail Bias | Mean-Gradient Cosine | Log-Std Cosine |
| --- | ---: | ---: | ---: | ---: |
| 50 / zero_train | 511.18 -> 115.11 | +44.37 -> -0.008 | .642 -> .737 | .491 -> .315 |
| 50 / joint_ppo | 411.99 -> 124.74 | +37.98 -> -1.29 | .630 -> .542 | .317 -> .681 |
| 100 / zero_train | 544.75 -> 226.24 | +41.81 -> -0.50 | .702 -> .631 | .599 -> .877 |
| 100 / joint_ppo | 512.80 -> 200.35 | +38.35 -> -1.82 | .737 -> .558 | .557 -> .711 |

Tail MSE falls97.6-98.8% in the four group means. Sign disagreement improves in all four, but mean-gradient alignment falls in three groups; root310037/period50/joint_ppo reverses to cosine-.083. Log-std alignment worsens in50/zero_train. Better value fit alone does not authorize another reward trial.

Next isolate episode-to-episode GAE/MC gradient reliability and the effect of changing the fitted baseline in the MC reference. Use all35 unordered balanced4/4 episode partitions, no chosen split or lambda sweep. This is read-only diagnosis, not a replacement performance gate. All comparisons remain descriptive; no reward/frequency/generalization claim. Cost:4352 archives,5,222,400 lower reconstructions,20,480 value steps per treatment,zero native or actor steps. [Compact statistics](../results/pointmaze_horizon_value_stage67_full_20261001_r1/compact_summary.json).
