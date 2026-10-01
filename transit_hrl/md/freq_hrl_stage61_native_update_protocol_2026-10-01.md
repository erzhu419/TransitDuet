# Stage61 One-Update Native Evaluation

Stage60 established nonzero lower movement under batch-mean conditional KL0.02, not reward gain. Reconstruct its exact first-update policies and evaluate clone/plain PPO/rejection-only/backtracking in the native PointMaze control loop. Keep the same Stage55 clone, Stage57 critic warmup/training batch, native credit, PPO settings and execution rules. Stage61 archive comparisons and costs must equal Stage60 exactly. No upper critic repair or further training is included.

Freeze roots310011,310023,310037,310049,310061,310073,310089,310101; periods50/100; zero_train/joint_ppo. Root310001 is preflight only. Use16 fresh common deterministic evaluation paths per full root,2 in preflight. Seed windows61_090_000/61_100_000 plus5001, with10000 offsets by full root; these are disjoint from source fitting, labels, training and prior evaluation. Clone executes zero residual and is evaluated once per period, shared across arms; each trained treatment retains its own arm's native upper execution. No root/period/path/checkpoint selection or reward-based tuning.

Full native budget:1792 episodes,2150400 primitive/lower calls,32256 upper calls and1792 trace audits. Archive reconstruction is additional, not free:4352 episodes,5222400 lower/78336 upper calls,1024 critic warmup updates,144 observed updates and their diagnostics, plus original optimizer steps and all guarded retry costs. Reuse the loaded forecaster and the same eight worker pool; no extra forecaster fitting, BC or evaluation gradient steps. Save96 candidate checkpoints and raw traces on the server only; do not pull them locally.

Primary metric is native episode return. Freeze12 paired contrasts: backtracking minus plain/rejection-only/clone for each period and arm. Average paths within root, then equal-weight the eight roots;65536 paired-root bootstrap draws, seed(61,61061), two-sided percentile intervals with Bonferroni12 family correction. Repair gain requires positive intervals versus plain in all four arm/period combinations; training gain separately requires positive intervals versus clone in all four. Negative and inconclusive endpoints remain visible; preflight cannot establish a performance claim.

Tests detect callback-induced archive changes, unpaired seed paths, duplicate clone cost, native execution/count errors, checkpoint metadata mistakes and bootstrap/gate errors. All tests, preflight, qualification and full tasks go through scheduler, dynamically on node001-node006 without hard pins. Full tasks9CPU/12GB; tests/preflight/qualification2CPU/4GB. Commit source/protocol and preregistration before full outcomes. Only logs and compact qualification JSON are local.

## Limitations

This is one-update development validation on reused teacher-initialized training roots with new evaluation paths. It is not full-training, frequency-specific, OOD, population-KL or equal-FLOPs confirmation. Existing Stage57 performance gates are unchanged; Stage61 outcomes govern only the named first-update comparisons.

## Execution

Code/protocol committed at15339d6843; test-only assertion correction at07917c7756. Task`t116482` passed16 existing regressions and3 new tests; the pipeline test failed because its state comparison included string configuration. Task`t116486` reran the stale remote test because tests were absent from its staging inputs. With corrected assertions and explicit tests staging, task`t116493` passed all4 new tests in54.018s on node006, exit0. No algorithm changes were needed; failed tasks remain recorded. Native preflight`t116497` registered under`pointmaze_native_update_stage61_preflight_20261001_r1` with source revisioncd2ea6c447; no performance outcome yet.
