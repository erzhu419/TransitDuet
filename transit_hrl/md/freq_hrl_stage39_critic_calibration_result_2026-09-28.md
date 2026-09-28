# Stage-39 Critic Calibration Result

Tasks `t102383`-`t102422`: all 40 cells/6400 evaluation episodes complete,
exit zero, without duplicate executions. Forty trajectory audits, 400 native
snapshot/mode replays, 80 probe/initial-credit replays and the independent
eight-endpoint bootstrap pass. Source `26dc18ab4f`; full registration `4080f198cd`.

| Deterministic return contrast | Mean | Eight-endpoint adjusted CI |
|---|---:|---:|
| Intrinsic-delayed vs frozen, final | -4.6238 | [-8.6284, 0.0753] |
| Intrinsic-calibrated vs frozen, final | -3.4395 | [-7.2946, 1.4610] |
| Task-delayed vs frozen, final | -3.4072 | [-7.9386, 1.0502] |
| Task-calibrated vs frozen, final | -3.0519 | [-7.4313, 1.1079] |
| Intrinsic calibration effect, final | 1.1843 | [-0.8009, 3.2260] |
| Task calibration effect, final | 0.3553 | [-2.1069, 2.7553] |
| Intrinsic calibration effect, first update | 0.0613 | [-1.3813, 1.5832] |
| Task calibration effect, first update | -0.1946 | [-1.6190, 0.7793] |

All eight endpoints are inconclusive. All four learned final return means
remain below frozen903.5155; lower-sampled final means also remain below
their frozen903.3726 reference. Critic calibration has no supported return gain.

At warmup16, fixed-probe mean MC value MSE falls from0.0109671 to0.0070719
for intrinsic and from567.3732 to208.9186 for task; actors remain unchanged.
Final fixed-probe Gaussian KL spans0.2030-0.2520 across learned arms.
Each learned arm charges5120 actor steps; critic steps5120 delayed/10240 calibrated.

Method cost:20064000 primitive steps/450660 upper/720194 gate calls.
Verification:576000 steps/12835 upper/20088 gate calls, charged separately.
Only [the compact summary](../results/pointmaze_critic_calibration_stage39_v1_full_20260928_r1/qualification_summary.json)
is local; raw trajectories and weights remain remote.

## Limitations And Next

Eight reused roots support conditional development, not independent confirmation.
One fixed stochastic probe per root is not exact expected-value truth; MSE/KL
are diagnostic. No noninferiority endpoint was registered, and a shorter actor
budget than Stage-38 does not establish a repair. Next isolate sampled frozen
upper/gate execution in training versus deterministic deployment, keeping
lower sampling, rewards, critic recipe and budgets matched. This mismatch is
a hypothesis, not an established cause. No new training jobs or retuning launched.
