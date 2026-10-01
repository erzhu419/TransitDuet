# Stage70: fixed state baselines did not reduce MC gradient variance

t117099-t117107 completed, exit 0; 69.0-76.6 seconds/root on node004-node006. Replayed all 1,024 Stage69 episodes without new native steps, fits, optimizer updates or checkpoints. Stage69 values/gradient metrics, covariance identities and model/Adam freeze checks passed. Twelve local tests passed. Only markers and 330 KB compact JSON were pulled.

Equal-root means for mean-parameter gradients; C = fixed Stage64 control and F = fixed Stage67 factored baseline. Ratios compare raw episode-gradient covariance traces with the unchanged common time-only baseline.

| Period / execution | MC-C/common variance | MC-F/common variance | MC-F raw within-batch-pair cosine | MC-F debiased mean SNR |
|---|---:|---:|---:|---:|
| 50 / zero_train | 6.050 | 1.123 | -0.018 | -0.213 |
| 50 / joint_ppo | 4.357 | 1.234 | -0.010 | -0.151 |
| 100 / zero_train | 3.371 | 1.136 | -0.051 | -0.264 |
| 100 / joint_ppo | 3.410 | 1.192 | -0.039 | -0.183 |

C reduced variance in 0/32 cases; F in 6/32. Better return prediction did not make F a useful unit-coefficient MC control variate. GAE-F covariance was only 4.3%-5.7% of MC-F covariance, but its SNR remained weak. Neither direct MC replacement nor actor adoption is justified; Stage67 HOLD remains unchanged.

## Limitations
These are descriptive diagnostics on already-used development paths and the discounted surrogate, not new independent confirmation or native reward improvement. Negative SNR estimates do not prove zero true gradient. A lower covariance trace alone is not proof of correct credit or a useful update; batch pairs are dependent.

Next: calibrate control variates against actor-gradient covariance on historical calibration archives, then freeze them before evaluating the Stage69 probes. Keep both critics, common reference and all failed evidence; no probe-fitted coefficients, lambda sweep or actor update. Value-MSE-only baseline selection is not the next direction.
