# Stage54 Result

Implementation `f4d58cd665`, audit-only correction `9f74619471`, full freeze `273f12e7e9`. First test attempt `t107773` failed on trace/history precision and a non-native cost-critic fixture; neither correction changes forecasting or control. All 26 regression tests passed in `t107781` (node006). Native preflight `t107777` and qualification `t107782` passed: 14400 steps, 48 audits. Full `t107783`-`t107790` covered node001-node006 and all completed with exit 0; qualification `t107795` passed on node005.

Full accounting: 3686400 native steps, 3072 audits, 55296 upper calls, 3686400 lower calls, zero gate/RL optimizer/extra verification steps. Forecast fitting: 256 disjoint driver paths, 307456 observations, 281600 rows/feature OLS fits, 28160000 displacement labels, eight ridge solves and zero native fitting steps. Execution/audit each used 34816 OLS fits and 17408 ridge predictions; 1228800 explicit actor-context evaluations; eight CARE solves. Local compact evidence retains all root endpoints, fitting counts and pooled arms/modes; raw traces, coefficients and full root means remain remote.

Native deterministic return differences, 65536 paired-root bootstrap draws and simultaneous Bonferroni12 intervals:

| Period | Contrast | Mean | CI | Effect |
| --- | --- | ---: | --- | --- |
| 50 | forecast main | 81.482 | [76.831, 86.158] | positive |
| 50 | velocity main | -0.838 | [-2.950, 0.817] | inconclusive |
| 50 | interaction | 93.170 | [87.004, 98.687] | positive |
| 50 | combined minus held | 79.117 | [73.396, 84.155] | positive |
| 50 | combined minus frozen | 49.966 | [43.412, 59.298] | positive |
| 50 | combined minus linear-position | 80.643 | [74.543, 86.045] | positive |
| 100 | forecast main | 100.641 | [92.494, 107.785] | positive |
| 100 | velocity main | -29.859 | [-34.914, -25.855] | negative |
| 100 | interaction | 61.035 | [55.843, 65.296] | positive |
| 100 | combined minus held | 24.411 | [13.328, 33.038] | positive |
| 100 | combined minus frozen | 51.338 | [37.559, 61.877] | positive |
| 100 | combined minus linear-position | 70.782 | [61.207, 79.101] | positive |

Decision: the preregistered combined candidate passes all six control contrasts at both periods. Forecast-reference error integral falls from linear's 1.058/4.469 to ridge's .311/1.325. The independent factorial separates the previously confounded mechanisms: linear velocity tracking loses 47.424/60.376 return versus linear position, whereas ridge velocity changes return by +45.747/+.658 versus ridge position. These last simple effects are descriptive, not additional tested endpoints; period100 does not establish incremental velocity benefit. Both deployment modes and all negative endpoints remain in evidence.

Next: use this positive mechanism as the reference for native learned-plan upper / learned lower training under the shared core and native task reward. Keep teacher, held and frozen controls, use fresh fitting/evaluation paths, and retain forecast/tracking ablations. Adoption must again use reward, not prediction MSE or an improved subset of periods.

## Limitations

This supports a learned forecaster with fixed analytic feedback under a conditional development protocol. The original learned upper is called but its action is ignored by the analytic-reference arms; the lower feedback is not trained by PPO. Consequently this is not learned Freq-HRL confirmation, frequency-separation attribution or an OOD result. Eight reused optimizer roots and simulator-generated fitting data require independent learned-policy confirmation; equal calls are not equal FLOPs.
