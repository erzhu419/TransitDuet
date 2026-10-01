# Stage73: paired native directional response

Stage72's objective-correct native MC reference remains unstable. Stage73 tests native forward reward responses rather than treating that finite-sample vector as truth.

Fit three fixed all-parameter directions on all original Stage57 warmup episodes: separately PPO-normalized control/factored GAE batch gradients with the inherited entropy coefficient, and raw native undiscounted MC with the first historical batch's time baseline and no entropy. Freeze each negative normalized loss-gradient direction before new native evaluation.

Use one nominal conditional Gaussian Fisher radius, delta=0.001. Functional JVP estimates curvature on historical states including the actor's actual std clamp. Both cloned policies use the same step `sqrt(2*delta/F)` with opposite signs. The historical mean of the two exact old-to-new KLs must lie in [delta/2, 2*delta]; otherwise stop the cell without radius tuning. Source networks, critics and all Adam states remain unchanged. Deliberate cloned-actor perturbations are counted, not called frozen source actors or optimizer updates.

Full: eight existing roots, periods 50/100 and zero/normal execution, 128 historical episodes and 32 fresh environment seeds per case. Run base and all six signed variants with the same initial policy RNG, stepwise lower noise and fixed upper proposals. Preflight: root 310001, four historical/four fresh episodes per case, H=300. New evaluation uses the 73M seed namespace, disjoint from calibration and Stage69 probes. Both actors sample, matching the historical execution distribution.

Report all 36 reward endpoints: plus-minus, plus-base, minus-base for three directions in four cases. Use the existing equal-root percentile bootstrap convention, 65536 draws, two-sided Bonferroni36 intervals. A plus-minus response is not itself a gain over base. Do not select directions, radius, parameter heads or seeds from these results. No actor adoption or training; Stage67 HOLD remains.

Full budget: 4096 archived calibration episodes, 4,915,200 reconstructed lower calls; 7168 native episodes, 8,601,600 native steps; 8192 score forwards/40,960 backwards; 14,400 Fisher JVP chunks/28,800 exact-KL forwards; 192 explicit cloned-actor perturbations. Plan solver calls are recorded separately from native inference calls. Nine CPU/8-GiB root jobs use dynamic node001-node006 placement. Pull compact JSON/markers only; do not write native raw traces or candidate checkpoints.

## Execution

Preflight t118615/t118616 passed: 112 native episodes, all 12 radius checks and all 16 paired-seed checks; source networks and Adam states unchanged. Preflight response signs differ between periods, so the full protocol and radius remain unchanged.

Full run `pointmaze_native_direction_stage73_full_20261001_r1`: t118625-t118632 launched; t118633 waits for all eight completion markers. Preflight/full resource histories are separated because the scheduler had applied the three-worker preflight RAM estimate to nine-worker full tasks; full declarations restored to 8 GiB.

## Limitations

Finite-radius stochastic policy responses are not infinitesimal derivatives or full learning curves. Teacher-initialized development roots and historical fits are reused; new evaluation paths are independent of direction fitting. Eight-root bootstrap uncertainty is limited, and this is not OOD or frequency-superiority evidence.
