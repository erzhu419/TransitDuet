# Stage71 result and next step

All eight roots and qualifier t118557-t118565 completed with exit code 0. Mechanical accounting passed: 4096 historical and 1024 probe episodes; 64 historical scalar fits; 32 Stage70 reproductions; 80 model/Adam freeze checks. No native steps, actor/critic optimizer updates or checkpoint writes.

| Period / arm | Calibrated control variance / common | Calibrated factored variance / common | Factored mean SNR | Factored raw batch cosine |
| --- | ---: | ---: | ---: | ---: |
| 50 / zero_train | 1.00360 | 0.97757 | -0.29939 | -0.06459 |
| 50 / joint_ppo | 1.00379 | 0.98551 | -0.18551 | -0.02217 |
| 100 / zero_train | 1.00001 | 0.99075 | -0.29313 | -0.05758 |
| 100 / joint_ppo | 1.00481 | 1.00361 | -0.32785 | -0.08370 |

These are equal-root means of mean-parameter diagnostics. Factored calibration reduces variance in 18/32 cases, but only 4/32 have positive debiased mean SNR and 7/32 positive raw batch repeatability. Historical coefficient fitting does not establish a stable actor signal. Retain this negative result and Stage67 HOLD; no coefficient/head/seed sweep.

Next: audit the learning objective before another policy change. Compare native undiscounted episode-reward score gradients, correctly time-weighted discounted episode gradients, and the existing uniform-time discounted-return surrogate on the same frozen paths. Keep critic weights, lambda and policies fixed.

## Limitations

Reused teacher-initialized development roots and Stage69/70 probes; descriptive covariance diagnostics, not independent confirmation, native reward improvement, or frequency superiority.
