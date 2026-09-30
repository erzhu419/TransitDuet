# Stage57 Result

Implementation `6a435ce0e6`; protocol/seed freeze `e673634741`. Tests `t108805` passed 39 tests; native preflight `t108833` and qualification `t108836` passed. Full tasks `t109115`-`t109122` all completed; full qualification `t109157` passed with exit 0 on node005. All eight roots, both periods, all three arms and both deployment modes retained; no training or protocol changes after outcomes.

Actual incremental cost: 16588800 native steps (4915200 critic calibration, 9830400 PPO training, 1843200 evaluation), 13824 native trace audits, 248832 upper and 16588800 lower calls. Actual optimizer steps: upper actor/value 2048/4096, lower actor/value 40960/61440. Reused 16 clone checkpoints/eight forecasters; zero new fits, supervised or extra verification steps. Upstream Stage55 remains separately charged at 16128000 native steps. Only compact JSON was pulled; full histories, traces and weights remain remote.

Native deterministic return, 65536 equal-root paired bootstrap draws, simultaneous Bonferroni6 intervals:

| Contrast | Period50 mean [CI] | Period100 mean [CI] |
| --- | --- | --- |
| joint_ppo - zero_train | +44.753 [22.416, 72.233], positive | +33.061 [18.608, 49.150], positive |
| joint_ppo - clone | +13.251 [-3.644, 29.199], inconclusive | +20.053 [12.894, 28.440], positive |
| zero_train - clone | -31.503 [-74.679, -2.024], negative | -13.008 [-27.985, 0.089], inconclusive |

Decision: **matched-upper gate passed; training-gain gate failed**. Joint still beats a lower trained with zero residual from its first calibration/training rollout, so Stage56's benefit is not confined to deployment-only deletion. However, the period50 control deteriorates and joint's improvement over the unchanged clone is not confirmed there. Absolute returns (clone / zero_train / joint): 956.899 / 925.396 / 970.149 at period50; 855.800 / 842.792 / 875.853 at period100. This is conditional evidence for the joint execution/training package, not uniform training improvement.

At period50, joint-clone is negative for roots 310049 (-20.773) and 310073 (-3.831); root 310049 zero_train-clone is -121.509. At period100 every joint-clone root mean is positive. Both arms changed lower actor/value parameters; only joint changed upper actor/value after calibration. Sampled-lower joint-clone differences are descriptively +12.755/+19.760, not substitute primary endpoints. These summaries identify a training-stability question; parameter movement alone does not diagnose its cause.

Next: instrument actual on-policy KL, clipping and critic-fit diagnostics, then preregister one justified stability intervention shared by both learned arms. Keep clone, both periods, matched lower/native budgets, fixed-final evaluation and fresh paths. Establish reliable clone-relative improvement before independently initialized training-root confirmation; retain both negative roots rather than selecting seeds or extending this completed experiment.

## Limitations

Eight reused training roots, teacher initialization and a fixed forecast make this development evidence. Lower optimizer/native budgets match, but joint has extra upper optimizer cost and total FLOPs do not match. This neither reverses Stage55's failed joint-gain gate nor establishes frequency-separation, learned timing, promotion, OOD or independent full-method confirmation.
