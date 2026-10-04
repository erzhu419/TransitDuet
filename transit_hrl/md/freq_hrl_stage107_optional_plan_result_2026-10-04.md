# Stage107 Optional-Plan Full Result

All9 tasks `t132282-t132290` are done/exit0 (scheduler archive). All8 roots pass the frozen protocol. Server-side source-cell reaggregation and an independently coded bootstrap from paired evaluation returns exactly reproduce all20 means and Bonferroni-corrected CIs. No checkpoints or raw trajectories were pulled.

| Primary reward contrast | Mean | Bonferroni20 CI | Positive roots |
| --- | ---: | --- | ---: |
| 50 learned hint minus blind | +0.004554 | [+0.002606,+0.007196] | 8/8 |
| 50 learned hint minus forecast hint | +0.000052 | [-0.000009,+0.000116] | 6/8 |
| 50 learned hint minus own blinded | +0.004629 | [+0.002309,+0.007679] | 8/8 |
| 100 learned hint minus blind | +0.007326 | [+0.006499,+0.009171] | 8/8 |
| 100 learned hint minus forecast hint | -0.000077 | [-0.000191,+0.000016] | 2/8 |
| 100 learned hint minus own blinded | +0.007143 | [+0.005825,+0.009071] | 8/8 |

Registered optional learned-plan confirmation: **not_supported**. Advice has repeatable but tiny value above strong flat feedback; learned upper advice has no confirmed increment over causal forecast. The family has14 positive,0 negative,6 inconclusive contrasts. These inconclusive results are not equivalence. Forecast minus blind is also supported:+0.004502/+0.007403 at50/100. Learned gains versus blind are about0.00040%/0.00064% of blind reward,not a practically large hierarchy advantage.

Mean blind/forecast/learned rewards:1152.225505/1152.230007/1152.230059 at50;1140.953632/1140.961034/1140.960957 at100. Blind itself improves over its donor base by+0.360099/+0.635365. Advice-trained lower policies with advice disabled are inconclusive versus the separately trained blind lower.

Exact new cost:52,224 native episodes/62,668,800 steps,384 lower-mean updates,304,128 native upper calls,574,464 OLS/ridge fits each,48 final server-only checkpoints. Native wall1527.24-1546.64s/root;sampled process-tree peaks5118-5413MiB. The complete inherited Stage106 campaign and Stage96/97 preparation are separate costs. See [compact evidence](../results/pointmaze_optional_plan_stage107_full_20261004_r1/compact_summary.json).

Next:fixed-policy crossed execution. Keep each learned/forecast lower fixed and swap forecast,learned residual,zero-mean same-std upper noise,and blind advice on fresh paired paths. This separates learned upper content from different lower weights and random residuals;no new training,alpha tuning,task change or weaker feedback. Stage106 negative evidence and Stage67 critic HOLD remain unchanged.

Follow-up complete: [Stage108](freq_hrl_stage108_crossed_advice_result_2026-10-04.md). Fixing the lower and matching upper innovations still yields no confirmed learned increment over forecast/noise. Stage107 is retained separately,not pooled.
