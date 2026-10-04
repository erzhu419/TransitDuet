# Stage110 Native Directional Gain Result

Preflight t134593/t134594 and full t134596-t134604 are all done/exit0. All8 roots and both periods completed the frozen A/B paired matrix. Six focused tests passed. Independent reconstruction from server-side paired returns exactly reproduces288 root effects,36 endpoint means and72 corrected CI bounds.

Registered gain gate: **not_supported** at both periods. None of16 directional gains over forecast has a positive corrected CI;all8 slope CIs cross0. Across the full36 family:3 positive,5 negative,28 inconclusive endpoints.

| Period | Forecast mean return | Learned-mean minus forecast | Bonferroni36 CI | Eligible directions |
| --- | ---: | ---: | --- | --- |
| 50 | 1151.396786 | -0.000500017 | [-0.001392644,-0.000113838] | none |
| 100 | 1140.168066 | -0.000154448 | [-0.000505461,+0.000065843] | none |

Forecast advice still improves on blind under the same fixed lower:+0.004310708/+0.006511937,both corrected CIs positive. The remaining positive endpoint is100/axis3 curvature;positive curvature is not a directional gain. The period50 learned-mean decrement is statistically resolved but practically tiny,about0.000043% of forecast return.

Exploratory root-specific A/B slope-vector alignment is positive in6/8 roots at50 and3/8 at100;mean cosine0.578/0.116. This does not replace the registered gate. [Compact results](../results/pointmaze_action_gain_stage110_full_20261004_r1/compact_summary.json),[independent reproduction](../results/pointmaze_action_gain_stage110_full_20261004_r1/aggregation_reproduction.json),[exploratory root gradients](../results/pointmaze_action_gain_stage110_full_20261004_r1/root_gradient_diagnostics.json).

Cost:3,072 native episodes/3,686,400 steps,41,472 upper calls,47,872 forecast OLS/ridge predictions each. All256 zero/forecast command identities and2,816 pairs per feedback/exogenous/noise check passed;max lower-innovation mismatch1.431e-6. No policy updates,critic fits,checkpoint or trace writes. Native wall146.57-151.67s/root after source loading;sampled process-tree RAM2295-2358MiB. Source preparation remains separate. Scheduler dynamically used node001/004/005/006;qualification/pref RAM unmeasured. [Terminal roster](../results/pointmaze_action_gain_stage110_full_20261004_r1/scheduler_tasks.json).

## Next

Close the old frozen-upper scale/bias route as a positive hierarchy contribution. Change the lower's trainable advice path:preserve all8 strong blind/flat donor functions as a common frozen base,and add a zero-initialized optional action-residual branch. Blind/forecast/learned arms must have the same architecture,samples and update budget;the base feedback is not weakened. Qualify paired native option credit on independent batches before learning the upper policy. Use a new development protocol,no pooling or further alpha/seed rescue.

## Limitations

These are deterministic whole-episode upper-bias interventions,not local state-conditioned Q values or a newly trained HRL algorithm. The negative gate does not prove equivalence or rule out state-specific/root-specific learnable gain. Exploratory alignment is descriptive,not confirmatory. Stage67 critic HOLD and Stage108's negative hierarchy boundary remain unchanged.
