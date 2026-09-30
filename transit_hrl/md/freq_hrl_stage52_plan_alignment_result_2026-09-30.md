# Stage52 Result

Recovery run `pointmaze_plan_alignment_stage52_full_20260930_r2` completed all eight unchanged roots (`t107175`-`t107182`, exit 0). Server qualification `t107194` on node004 passed. Compact evidence: `results/pointmaze_plan_alignment_stage52_full_20260930_r2/qualification_summary.json`.

All eight preregistered deterministic return endpoints use equal-root paired bootstrap, 65536 draws and two-sided Bonferroni8 intervals:

| Period | Contrast | Mean | CI | Effect |
| --- | --- | ---: | --- | --- |
| 50 | held target - waypoint | -16.642 | [-21.975, -11.431] | negative |
| 50 | curve - held target | -1.441 | [-4.454, 1.782] | inconclusive |
| 50 | curve - reverse | 18.747 | [11.940, 25.948] | positive |
| 50 | curve - frozen | -39.731 | [-49.304, -30.637] | negative |
| 100 | held target - waypoint | 18.368 | [13.368, 24.571] | positive |
| 100 | curve - held target | -46.022 | [-50.564, -40.660] | negative |
| 100 | curve - reverse | -27.646 | [-37.638, -16.910] | negative |
| 100 | curve - frozen | -25.072 | [-36.518, -15.364] | negative |

Decision: reject this linear reference-curve candidate. Neither period satisfies the frozen curve-versus-held/reverse/frozen adoption gate. Do not select period50's positive reverse-control contrast or period100's held-target contrast as overall plan utility.

Descriptive diagnostics: reference-target squared-error integrals are held/curve/reverse = 0.686/1.038/1.515 at period50 and 2.023/4.573/3.738 at period100. Past-slope extrapolation worsens target-reference error versus holding at both periods; at period100 it also loses to reverse. Current-target feedback remains the best absolute return (947.115 at either period), versus frozen 918.004/821.517 and curve 878.273/796.445. Sampled execution has the same descriptive ordering. These observations motivate a target-forecast/control interaction diagnosis, not another gain, window or optimizer search.

Verified recovery cost: 3686400 native steps, 3072 audits, 55296 upper calls, 3686400 lower calls, 17408 plan fits and 17408 audit fits; eight CARE solves, zero gate calls, actor/value optimizer updates or extra verification simulation. First-attempt accounting is retained in `results/pointmaze_plan_alignment_stage52_full_20260930_r1/attempt_status.json`: 1516800 reconstructed native steps. Combined attempt cost is 5203200 steps. Only the 78 KB compact summary was pulled; raw trajectories and checkpoints remain remote.

Next: use the recorded exogenous paths to separate forecast error at turns/regime changes from controller tracking error before choosing a new plan class. A subsequent learned-upper experiment must evaluate its own native task return against held-reference and fixed-clock baselines; it must not inherit success from the current-target diagnostic. No Stage52 window/gain/root/period/mode retuning or seed extension.

## Limitations

This is an analytic plan-class diagnostic with fixed controllers, not learned Freq-HRL validation or proof against all plan representations. Roots are reused conditional-development sources, not independent confirmation. Equal inference call counts do not imply equal FLOPs; regression cost is counted. Stage51's conclusions remain unchanged.
