# Stage72 result and next step

All eight roots completed, and scheduler qualification t118585 passed. Work: 1024 probe and 256 historical archive episodes; 128 native-return identities; 32 Stage70 reproductions; 80 model/Adam freeze checks. No native environment steps, optimizer updates or checkpoint writes.

| Period / arm | Native MC mean SNR | Native MC batch cosine | Control GAE mean SNR | Control GAE cross-independent native cosine |
| --- | ---: | ---: | ---: | ---: |
| 50 / zero_train | -0.09568 | -0.04222 | 0.27748 | 0.03774 |
| 50 / joint_ppo | -0.02383 | -0.04333 | 0.26466 | -0.00950 |
| 100 / zero_train | -0.16268 | -0.05455 | 0.48321 | 0.08252 |
| 100 / joint_ppo | -0.35218 | -0.11127 | 0.36521 | 0.03500 |

Equal-root means of mean-parameter diagnostics. Native MC has positive SNR in 13/32 cases and positive batch repeatability in 10/32; control GAE has positive SNR in 22/32. The native time baseline reduces native zero-baseline variance to 0.27%-0.63%, but the remaining native MC reference is still unstable. Correctly time-weighted discounted MC also has negative group-mean SNR and repeatability. Changing the objective formula did not establish a reliable actor direction; noisy MC cosines cannot validate or reject native reward gains. Stage67 HOLD remains.

Next: fit fixed directions on historical calibration only, then measure paired native reward responses to symmetric small cloned-policy perturbations on fresh seeds. Keep source policies, critics and optimizers unchanged; count every deliberate parameter perturbation and native rollout. No coefficient, radius, direction-head or seed selection after results.

## Limitations

Teacher-initialized development roots and Stage69 probes are reused. These are descriptive objective/noise diagnostics, not independent performance confirmation or frequency superiority.
