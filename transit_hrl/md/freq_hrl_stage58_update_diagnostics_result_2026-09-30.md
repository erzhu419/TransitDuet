# Stage58 Update Diagnosis

All eight replay tasks `t109185`-`t109192` and qualification `t109196` completed with exit 0. Archived upper/executed lower actions and final four networks/four Adam states matched exactly at both periods in both learned arms. Replayed 12288 episodes, 14745600 lower/221184 upper calls and 2560 updates; optimizer steps upper actor/value 2048/4096, lower actor/value 40960/61440. New native/evaluation/fit counts are zero. The compact JSON retains every root and selected iteration diagnostics; traces and weights remain remote.

## Findings

Equal-root means over all training updates; final EV is against each final batch's own GAE targets, not held-out prediction quality.

| Period | Arm/level | Conditional KL/update | Clipped samples | Final value EV | Final value MSE |
|---|---|---:|---:|---:|---:|
| 50 | zero_train/lower | 0.2313 | 31.43% | 0.9874 | 1.314 |
| 50 | joint_ppo/lower | 0.2083 | 30.44% | 0.9835 | 1.607 |
| 50 | joint_ppo/upper | 0.0259 | 30.43% | -0.00022 | 473.135 |
| 100 | zero_train/lower | 0.0199 | 23.46% | 0.9876 | 3.564 |
| 100 | joint_ppo/lower | 0.0206 | 24.18% | 0.9877 | 3.452 |
| 100 | joint_ppo/upper | 0.0214 | 27.14% | -0.00007 | 1066.683 |

The period50 lower means are dominated by first-update displacement: root310049 zero_train KL=43.0505, clipping=98.29%, action-mean RMS change=1.1669; root310073 joint_ppo KL=45.5558, clipping=99.53%, RMS=1.2321; root310073 zero_train KL=7.4694, clipping=95.41%. Their next sampled training-return means fell 971.824 -> 840.583, 806.380 -> 606.272 and 944.258 -> 845.671 respectively. These adjacent batches use different paths. Root310049 joint_ppo also lost against clone without such a KL spike, so overshoot does not explain every failed endpoint.

Upper value EV was already near zero after critic-only calibration (joint equal-root EV 0.00164/0.00125 at periods50/100), and remained so during learning. Lower calibration improved EV from 0.168/0.336 to 0.929/0.962. Before-update log-probability errors stayed tiny (maximum lower 3.20e-5, upper 3.82e-6); no action/log-probability reconstruction mismatch was found.

## Decision And Next Step

Keep Stage57's matched-upper pass and clone-relative training-gain failure unchanged. First freeze an archive-only first-update comparison of unchanged PPO versus full-batch conditional-KL-controlled actor updates, restoring both actor and Adam state on rejected steps. Apply the same rule to both learned arms and periods, with unchanged credit, critic updates and initialization; count extra distribution checks and rejected work. Use conditional KL per decision, not Stage48/49's near-frozen whole-episode budget. Native performance validation follows a valid fixed-batch test; upper critic calibration/representation is the next separate intervention. No reward-based tuning, Adam reset or new seed expansion is justified by this diagnosis.

## Limitations

These are post-hoc training-batch diagnostics on reused development roots. Temporal association is not causal attribution, conditional Gaussian KL is not an environment-trajectory bound, and critic EV uses bootstrapped targets. No new performance, frequency-specific, promotion, OOD or submission-readiness claim follows.
