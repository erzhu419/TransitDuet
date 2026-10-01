# Stage69 result: better values, unreliable gradient signal

Completed t117012-t117020, exit 0: eight roots, 1,024 fresh episodes, 1,228,800 native steps; 76.8-80.2 seconds/root on node004-node006. Accounting, historical probes, TD identities and model/Adam freeze checks passed. Eight local tests passed. No critic/forecaster fits, optimizer steps or checkpoints were added.

Equal-root descriptive means below use C = Stage64 mc_normalized and F = Stage67 mc_factored. Gradient cosines refer to mean parameters; independent batches contain eight episodes each.

| Period / execution | Fresh value MSE C -> F | Fresh TD MSE C -> F | Within-GAE cosine C -> F | Within-common-MC cosine | Common-MC debiased SNR, mean of 32 episodes |
|---|---:|---:|---:|---:|---:|
| 50 / zero_train | 538.59 -> 107.91 | 8.331 -> 0.102 | 0.049 -> -0.083 | -0.085 | -0.322 |
| 50 / joint_ppo | 430.44 -> 130.83 | 5.814 -> 0.168 | 0.084 -> 0.006 | -0.016 | -0.181 |
| 100 / zero_train | 588.98 -> 212.52 | 7.001 -> 0.123 | 0.108 -> 0.038 | -0.076 | -0.353 |
| 100 / joint_ppo | 516.37 -> 187.88 | 5.939 -> 0.132 | 0.084 -> 0.041 | -0.088 | -0.332 |

All four group means confirm better critic fit, including tail bias and renewal TD residuals, but not reliable update directions. F's same-batch GAE/common-MC cosine of 0.427-0.507 falls to -0.070 to 0.020 across batches; 28/32 cases have nonpositive common-MC debiased mean-gradient SNR estimates. Stage67 HOLD remains unchanged.

## Known Issues
Preflight r1/r2 exposed MC precision and string-config comparison errors; r3 fixes both while preserving historical probes. Historical MC rounding differences were at most 0.000386. Reused teacher-initialized policies, discounted surrogate and dependent batch pairs limit inference: negative SNR estimates do not prove zero true gradient or systematic GAE bias. There is no reward/frequency-superiority claim or new CI.

Next: reuse server-only fresh archives to compare raw MC-minus-fixed-causal-state-baseline gradients against GAE, keeping the common reference unchanged. Test variance reduction before a separately registered actor intervention, without new seeds, probe-fitted baselines or lambda tuning. Pulled only markers and 275 KB compact JSON; traces/weights remain server-only.
