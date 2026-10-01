# Stage72: native objective alignment

Stage71's historical scalar fitting did not establish a stable MC actor direction. Stage72 audits the objective rather than tuning coefficients. Source inspection confirms task reward reaches the lower rollout; the unresolved difference is credit weighting.

For fixed horizon H, the raw loss-gradient estimators are:

- Native episode reward: `-sum_t score_t * (sum_{u>=t} r_u - b_native(t)) / H`.
- Discounted episode reward: `-sum_t score_t * gamma^t * (sum_{u>=t} gamma^(u-t) r_u - b0(t)) / H`.
- Existing surrogate: the same discounted reward-to-go, but without `gamma^t`. Existing native-lambda GAE control/factored critics are retained as controls.

`b_native(t)=(H-t)*mu`, with mu the mean task reward of the first historical Stage57 warmup batch only. Also report a zero-baseline native estimator. Reproduce all five Stage70 control estimators. Use the existing eight roots, both periods and execution arms, all four Stage69 probe batches of eight episodes at H=1200. Preflight is root 310001, H=300, two historical and four probe episodes per case. No new policy, critic or coefficient fit; no gamma/lambda/seed sweep.

Primary diagnostics: raw independent-episode covariance, signed debiased SNR, batch repeatability and cross-independent-batch native/discounted direction cosines. Separately normalized PPO directions are secondary. Variances of different objectives are not variance-reduction comparisons; only native fixed-baseline versus native zero-baseline shares an objective. An exact two-step Gaussian quadrature test must distinguish the three objective gradients before server submission.

Full budget: 256 historical + 1024 probe archive episodes; 1,536,000 reconstructed lower calls; 23,040 upper calls; 2048 score forwards and 20,480 backwards; 32 historical reward-rate frames. No new native steps, optimizer updates or checkpoint writes. Scheduler uses five CPU/4 GiB root jobs, dynamic node001-node006 placement, and a one-CPU qualifier. Pull compact JSON/markers only.

## Limitations

Teacher initialization and development probes are reused. Correct objective algebra does not imply the finite-sample native MC direction is reliable or that replacing PPO's discount improves performance. Retain negative signal estimates and Stage67 HOLD; this audit alone cannot authorize actor adoption or establish frequency superiority.
