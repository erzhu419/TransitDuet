# Stage61 Native First-Update Result

Full tasks `t116526`-`t116533` and qualification `t116546` completed with exit0. All eight roots, both periods, both training arms and all treatments retained. Exact Stage60 update/critic reproduction, common fresh paths, native execution and frozen cost checks passed. **Repair-gain gate failed; clone-relative training-gain gate failed.** All12 preregistered corrected intervals cross zero.

Native deterministic episode-return differences, backtracking minus comparator; equal-root paired bootstrap with65536 draws and Bonferroni12 simultaneous intervals:

| Period | Arm | Versus Plain PPO: mean [CI] | Versus Rejection-Only: mean [CI] | Versus Clone: mean [CI] |
| --- | --- | --- | --- | --- |
| 50 | zero_train | +26.426 [-2.248, +86.712] | -1.118 [-2.288, +0.474] | -1.118 [-2.288, +0.474] |
| 50 | joint_ppo | +37.574 [-4.339, +156.048] | -0.800 [-2.365, +0.710] | +2.549 [-3.156, +7.529] |
| 100 | zero_train | +0.150 [-2.225, +2.756] | -1.086 [-3.236, +1.090] | -1.086 [-3.236, +1.090] |
| 100 | joint_ppo | +1.352 [-1.004, +4.117] | -0.045 [-2.562, +1.894] | +0.312 [-3.153, +3.437] |

The large period50 plain-relative means reflect recovery from a few extreme drops, not consistent gains: root310049 zero_train gains142.830 and root310073 joint_ppo gains310.855. Backtracking beats plain in5/8 zero_train and3/8 joint roots at period50, and3/8 and5/8 at period100. Against clone, only2/8 zero_train roots improve at either period; joint improves in5/8. No roots are removed. Rejection-only equals clone in zero_train because the lower actor is frozen and upper residuals execute as zero; its identical contrasts are expected, not independent confirmation.

Actual new native cost:1792 episodes/audits,2150400 primitive/lower calls,32256 upper calls,30464 plan OLS/ridge and matching audit operations,96 candidate checkpoints. Exact reconstruction is charged separately:4352 archived episodes,5222400 lower/78336 upper calls,1024 warmup critic updates and144 observed updates; Stage60 optimizer and all guard/retry costs remain in the compact record. No new fits, supervised steps or extra evaluation forecaster loads. Only58.1KB of compact JSON was pulled; raw traces/weights remain remote.

Next: retain backtracking as a valid displacement-control mechanism, not an adopted performance improvement. Before another native training matrix, diagnose lower advantage/option-boundary task credit and upper critic held-out fit using existing server archives, then preregister one independently tested intervention. Do not expand seeds, tune KL or reinterpret the completed gates. Stage55/57 decisions remain unchanged.

## Limitations

One update on eight reused teacher-initialized development roots, with fresh deterministic evaluation paths. These results neither establish nor rule out later training gains; they do not establish frequency-specific superiority, OOD generalization, full-training stability or equal-FLOPs benefit. CI crossing zero is inconclusive, not proof of no effect.

[Compact data](../results/pointmaze_native_update_stage61_full_20261001_r1/compact_summary.json).
