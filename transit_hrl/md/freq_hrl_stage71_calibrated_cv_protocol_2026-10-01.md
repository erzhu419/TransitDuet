# Stage71: historical score-covariance calibration

Stage70's unit-coefficient state baselines inflated raw MC actor-gradient variance. Stage71 tests whether a historical actor-covariance coefficient generalizes to fixed independent archives, with both policies and critics frozen.

For each root/period/execution arm and each fixed critic, fit one signed scalar on all original Stage57 warmup episodes: `alpha = trace Cov(g0,h) / trace Var(h)`, where `h = g0 - g_MC_state`. Use all actor parameters; no clipping, grid, probe fitting, or separate mean/log-std coefficient. Exact zero baseline sample variance gives alpha zero. Candidate signal is `MC - b0 - alpha*(V - b0)`. The common time-only baseline, discount, horizon, native GAE lambda and critic weights remain unchanged. Critic value inputs precede the current lower action.

Freeze alpha before reading Stage69 probe labels. Full: eight existing roots, periods 50/100, zero_train/joint_ppo; 128 calibration episodes and four probe batches of eight episodes per case, horizon 1200. Preflight: root 310001, four calibration and four probe episodes per case, horizon 300. Reproduce all Stage70 probe noise/direction controls. Test coefficient mathematics, streaming covariance, direct score-gradient linearity, seed separation and scheduler budgets before submission.

Primary outputs: raw episode covariance, signed debiased mean SNR, independent-batch direction repeatability and common-reference alignment. Separate per-batch PPO normalization is secondary. Report every root and both critics, including variance inflation and negative signal-power estimates; calibration-optimal variance is not held-out performance.

Full budget: 4096 historical + 1024 probe archive episodes, 6,144,000 reconstructed lower calls, 92,160 upper calls, 10,240 score forwards, 59,392 score backwards, 64 scalar fits. No new native steps, critic fits, optimizer steps or checkpoint writes. Scheduler dynamically places five-CPU/4-GiB root jobs on node001-node006; qualifier is one CPU/1 GiB. Pull completion markers and compact JSON only.

## Limitations

Critics and coefficients share historical calibration data; Stage69/70 probes and teacher-initialized development roots are reused. This is a diagnostic of a uniform-time discounted-return surrogate, not native reward, new confirmatory evidence, or frequency superiority. Stage67 HOLD remains unchanged regardless of this diagnostic. If coefficients fail to generalize, retain that result rather than sweep coefficients on these probes.
