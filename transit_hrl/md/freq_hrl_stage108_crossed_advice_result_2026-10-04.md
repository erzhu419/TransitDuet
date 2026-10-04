# Stage108 Fixed-Lower Learned Content: Full Result

All9 tasks `t134562-t134570` done/exit0. All8 roots pass the frozen execution protocol. Source-cell reaggregation and an independent equal-root bootstrap from paired evaluation returns exactly reproduce all20 means and Bonferroni-corrected CIs. No checkpoints or raw traces were pulled.

| Primary fixed learned-lower reward contrast | Mean | Bonferroni20 CI | Positive roots |
| --- | ---: | --- | ---: |
| 50 learned plan minus forecast | -0.000032095 | [-0.000083701,+0.000006414] | 2/8 |
| 50 learned plan minus same-std noise | -0.000006737 | [-0.000019306,+0.000002134] | 2/8 |
| 100 learned plan minus forecast | -0.000042139 | [-0.000114396,+0.000027040] | 1/8 |
| 100 learned plan minus same-std noise | -0.000002028 | [-0.000014829,+0.000008921] | 4/8 |

Registered learned-residual confirmation: **not_supported**. All4 primary intervals cross0;their point estimates are negative. The full20 family contains8 positive,0 negative,12 inconclusive contrasts. The forecast-trained-lower cross reaches the same boundary:learned-minus-forecast -0.000031850/-0.000043336 and learned-minus-noise -0.000006733/-0.000001824,both periods inconclusive.

Both forecast and learned advice still beat blind execution under both fixed lowers. On the learned lower,forecast-minus-blind is+0.004553/+0.006544,learned-minus-blind +0.004521/+0.006502. The stable tiny advice value from Stage107 remains,but the learned upper's output has no confirmed increment over a causal forecast or a same-std zero-mean residual. This is not equivalence or universal HRL invalidity.

New cost:4,096 episodes/4,915,200 steps,36,864 upper calls,52,224 OLS/ridge fits each. Zero policy updates,critic fits,new checkpoints or traces. Max upper-innovation pairing mismatch4.768e-7. Native wall224.37-229.45s/root;sampled process-tree peaks2308-2341MiB,qualification RAM unmeasured. Donor preparation and Stage106/107 training remain separate inherited costs. See [compact evidence](../results/pointmaze_crossed_advice_stage108_full_20261004_r1/compact_summary.json).

Next:close these mandatory/advisory frozen-donor routes as current positive hierarchy contributions. Change the upper learning problem before another reward matrix:measure controllable,action-conditioned gain over the intact flat lower and forecast,then train a plan-validity/replan value only if that gain exists. Re-express control-response calibration on the current lower rather than inherit the old waypoint-clone bound. Preserve strong flat feedback,all donor roots and negative evidence;any changed objective/decoder needs a separate development protocol,not rescue pooling. Stage67 critic HOLD is unchanged.
