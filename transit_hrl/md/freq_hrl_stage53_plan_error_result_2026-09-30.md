# Stage53 Result

Implementation `e6cc55d75e`, diagnostic freeze `b96395bd00`. Five focused tests passed in `t107755` on node005. Offline diagnosis `t107756` on node001 completed with exit 0: 3072 raw traces, 3686400 recorded steps and 128 driver regenerations; zero new native steps or optimizer updates. Local evidence is the 47 KB `results/pointmaze_plan_error_stage53_full_20260930_r1/compact_summary.json`; full summaries and raw trajectories remain remote.

Curve-minus-held error integrals, equal-root paired bootstrap with 65536 draws and Bonferroni8 intervals. Positive differences increase the indicated term; the cross term is signed:

| Period | Term | Mean | CI |
| --- | --- | ---: | --- |
| 50 | forecast | 0.352244 | [0.313561, 0.382292] |
| 50 | controller/reference | 1.183332 | [1.124422, 1.246120] |
| 50 | cross | -1.667680 | [-1.743781, -1.591574] |
| 50 | total vector error | -0.132104 | [-0.181112, -0.085988] |
| 100 | forecast | 2.550465 | [2.367612, 2.682835] |
| 100 | controller/reference | 2.984368 | [2.832412, 3.141016] |
| 100 | cross | -4.699694 | [-4.910843, -4.506238] |
| 100 | total vector error | 0.835139 | [0.738070, 0.901215] |

The forecast and controller/reference mismatch both worsen at both periods. Cross-term cancellation explains why period50's total squared error improves even though its Stage52 native reward contrast remains inconclusive (-1.441); reward is exp(-distance), not negative squared error. Therefore forecast MSE or integrated tracking error cannot replace native task reward as the adoption gate.

Descriptive timing decomposition of the return contrast: period50 history-only/future-only/both = -7.353/+6.308/-0.138, with clean -0.258; period100 = -5.824/-22.739/-17.459, with clean 0. Period100's early third accounts for -25.534 return, whereas most additional forecast error appears in the late third (1.941 of 2.550). This points to both stale-history prediction and carry-over controller/reference interaction, rather than a gain-only repair. Both sampled and deterministic modes and all event/phase partitions are retained in the compact evidence.

Decision: Stage52 remains rejected. Next intervention should separate a causal regime-aware forecast from planned-position/velocity tracking, using disjoint fitting and evaluation paths. Preserve held-reference/frozen controls and use native reward as the primary gate. Do not retune Stage52's window, gain, period or deployment mode, and do not extend its root roster.

## Limitations

These are post-outcome mechanism diagnostics, not causal mediation or learned Freq-HRL confirmation. At period100, the 63-increment fitting history plus 99 future option increments exceeds the driver's maximum 160-step regime dwell; every post-initial option is regime-exposed. Consequently stable and geometry-only options at that period are initial zero-velocity null controls, not evidence of a successful steady-state forecaster. Event-stratum differences are descriptive; cross terms do not permit nonnegative responsibility percentages.
