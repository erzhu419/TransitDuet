# Stage58 Protocol

Stage57's matched-upper gate passed, but joint-clone at period50 remained inconclusive and zero_train deteriorated. Diagnose all eight roots, both periods and both learned arms without new environment sampling or a stability sweep. Reconstruct the existing archived warmup/training batches from physical/measurement/goal/reference/context/reward arrays and original policy-noise seeds. Preserve scalar inference, original PPO credit, shuffle order and optimizers. Require exact recorded upper actions, executed lower actions, and final four networks/four optimizer states before using the diagnostics.

Measure old-to-new conditional Gaussian KL and summed per-episode conditional KL, PPO clipping, importance-ratio error before updates, policy std, GAE target scale and critic MSE/explained variance before/after each actual update. Critic calibration must leave actors unchanged. These are training-batch diagnostics, not causal proof of why a root failed. An intervention is chosen only after reviewing them and is applied equally to both learned arms in a separate frozen performance experiment.

Full replay: 12288 archived episodes, 14745600 lower and 221184 upper inference calls; 2560 observed updates with 5120 distribution passes, 5120 value passes and 2560 extra diagnostic GAE calls. Replay the original optimizer path (upper actor/value 2048/4096; lower actor/value 40960/61440). Load 16 source clones, eight saved forecasters and 32 final reference checkpoints. Zero new native/evaluation/supervised steps or forecaster fits. Preflight: 32 archived episodes, 9600 lower/144 upper calls, 28 observed updates. Historical Stage55/57 costs remain separately declared.

Tests, replay and qualification use scheduler's dynamic node001-node006 pool. Full roots use eight reconstruction workers plus learner, nine CPU/12 GB each; preflight and qualification two CPU/4 GB. Pull compact JSON only; archive traces, final weights and iteration diagnostics remain remote. If action or final-state reconstruction fails, fix the reconstruction rather than relax identity or reinterpret its metrics.

## Limitations

This is post-hoc development diagnosis of the frozen Stage57 optimizer, not new performance confirmation. Summed conditional KL is an empirical trajectory-scale diagnostic, not a proven off-policy trajectory bound. No gate, encoder, frequency-specific, OOD or independent training-root claim is introduced; Stage55/57 decisions remain unchanged.
