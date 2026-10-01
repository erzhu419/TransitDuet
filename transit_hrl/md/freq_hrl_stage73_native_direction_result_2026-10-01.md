# Stage73: native directional response result

All eight root jobs t118625-t118632 and qualification t118633 finished with exit 0. The frozen audit completed 7168 native episodes / 8,601,600 native steps, 1024 paired-seed checks and 96 radius checks. Exact historical KLs were 0.00099848-0.00100152 for the nominal radius 0.001. Root runtime was 436-447 seconds; measured root RAM peaks were 4599-4678 MiB.

All 36 endpoints use equal-root means and the preregistered 65536-draw, two-sided Bonferroni-36 root-bootstrap intervals. There are four positive, two negative and 30 inconclusive endpoints. The two supported plus-base gains and their normal-execution counterparts are:

| Period | Execution | Direction | Plus-base reward | Adjusted CI | Result |
|---|---|---|---:|---|---|
| 50 | zero_residual | factored GAE | +0.3834 | [0.0852, 0.6533] | positive |
| 50 | normal | factored GAE | +0.0705 | [-0.0434, 0.1993] | inconclusive |
| 100 | zero_residual | native MC | +0.6112 | [0.1603, 1.0775] | positive |
| 100 | normal | native MC | +0.0673 | [-0.1719, 0.2619] | inconclusive |

For both supported zero-residual cases, plus-minus is positive and minus-base is negative. Plus-base is positive in 7/8 roots for factored GAE at period 50 and 8/8 for native MC at period 100. All 18 normal-execution endpoints remain inconclusive. Every endpoint and root effect is retained in `results/pointmaze_native_direction_stage73_full_20261001_r1/qualification_compact.json`.

This establishes localized finite-radius native improvement, not successful joint-HRL training. Source networks and Adam states stayed unchanged; the audit used 192 deliberate cloned-actor perturbations and no optimizer steps. Stage67 HOLD remains.

## Next Step

Cross historical direction-fitting arm with native execution arm, retaining all three directions and the fixed radius, on newly preregistered paired evaluation seeds. Stage73 changes both fitting data and execution between arms, so it does not identify which one explains the smaller normal-execution response. Resolve that distinction before adopting a direction or restarting joint training.

## Limitations

Teacher-initialized development roots and historical fits are reused. Eight-root percentile-bootstrap uncertainty is limited. These stochastic finite-radius probes are not infinitesimal derivatives, full learning curves, OOD validation or evidence of frequency superiority.
