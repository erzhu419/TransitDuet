# Stage62 Credit/Critic Result

Tests`t116561`, preflight`t116563`/qualification`t116565`, full tasks`t116567`-`t116574` and full qualification`t116587` completed with exit0. All eight roots, periods and arms retained;32 complete model/Adam states remained unchanged. No training or protocol changes after preflight.

Equal-root descriptive means. Sign disagreement uses normalized post-update advantages on the archived first training batch / held-out deterministic paths; it is not the original training gradient:

| Period | Arm | Held-Out Lower Option-MC EV | Option/Episode Sign Disagreement: Fit / Held Out | Held-Out Upper Prediction / MC Target Std |
| --- | --- | ---: | ---: | ---: |
| 50 | zero_train | 0.855 | 62.2% / 64.9% | 0.054 / 33.135 |
| 50 | joint_ppo | 0.854 | 55.7% / 68.1% | 0.050 / 33.271 |
| 100 | zero_train | 0.823 | 52.5% / 53.1% | 0.057 / 28.016 |
| 100 | joint_ppo | 0.828 | 50.7% / 53.4% | 0.051 / 28.024 |

Lower critics fit the finite-option objective reasonably well, but this objective differs substantially from continuing episode credit. Across all32 root/period/arm training groups, option/episode normalized sign disagreement ranges44.6%-76.6%; the equal-root means above are not driven by a single root. Trace-cut with continuing bootstrap is much closer to episode-continuing GAE (held-out correlation0.992-0.998, sign disagreement2.0%-2.5%). Artificial boundaries suppress mean bootstrap terms29.6/32.7 at period50 and43.4/44.5 at period100 (joint/zero); these values come from option-trained critics and are sensitivity measures, not valid continuing-value corrections by themselves.

Upper prediction is nearly constant on both fitting and held-out archives. All64 root/period/arm/split upper-MC EVs lie[-0.003524,+0.000280]. Held-out equal-root MC biases are-77.4 to-87.0; even fitting-batch GAE-target biases are-17.7 to-34.5. This is not merely an observed held-out fit gap. Existing upper credit, representation and calibration require a separate intervention; this diagnosis does not identify which one causes the native reward result.

Decision: next isolate lower native episode credit with a correspondingly recalibrated lower critic, keeping actors' initialization, upper credit/representation, plan/forecaster, KL0.02 backtracking, roots and periods unchanged. First test paired archived first updates and critic fit; preregister fresh native evaluation only after valid mechanics. Upper critic repair follows independently. No seed expansion, KL sweep or performance adoption follows from these diagnostics; Stage55/57/61 gates remain unchanged.

Actual incremental cost:768 archived episodes,32 checkpoint loads,921600 lower/13824 upper feature and value rows,2304 lower/768 upper forward batches,3072 GAE and2304 MC calls. New environment/actor/optimizer/fit/forecaster/checkpoint-write counts0. Only152.1KB of full compact statistics pulled; traces/weights remain remote.

## Limitations

Post-update value diagnostics on reused development roots, not original pre-update advantages or a causal credit ablation. The held-out deterministic policy differs from stochastic fitting trajectories; realized MC residuals mix policy/distribution effects and sampling noise. An option-trained critic is not a consistent episode-value estimator. No reward, frequency, promotion, OOD or submission-readiness improvement is established here.

[Compact data](../results/pointmaze_credit_diagnostics_stage62_full_20261001_r1/compact_summary.json).
