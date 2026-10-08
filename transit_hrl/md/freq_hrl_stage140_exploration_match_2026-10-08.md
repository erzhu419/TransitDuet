# Stage140: Exploration With Matched Mean Steps

Stage139 Fisher increments are positive under sampled upper at all four
root-periods, but mean deployment still fails one. Warm std0.15 loses2.58/4.97
return relative to its own mean at50/100. Compare std0.15/0.05, not more seeds.

Roots410037/410049, periods50/100, horizon1200. Eight fresh paired training
scenarios/two independent noise folds, with new single-option counterfactual
credits for each std. Reuse the same warm actor mean, strong lower, values,
forecaster and authority0.05. Existing standardized damped Fisher score only,
damping1, both update signs, no Adam/value optimization or native-return fit.

Hold mean-output RMS step at0.001767767. Fixed std0.15 KL0.000555556 becomes
KL0.005 at std0.05. Equal KL would shrink the reduced-std mean step threefold;
that would mix smaller policy updates with the exploration comparison.

Evaluate32 new paired scenes/period under mean and sampled upper. Forecast is
executed once; warm mean is identical across stds and executed once. Sixteen
table entries/scene come from12 actual episodes and4 explicitly counted aliases.
Separate reduced-minus-original return from reduced-minus-original learning
increment, so less exploration loss is not confused with better learning.

Per root64collection+1,152counterfactual+768evaluation=1,984episodes/2,380,800steps;
two roots3,968episodes/4,761,600steps.256table aliases/root; all full-prefix query
costs count. Sixteen workers+parent/12GiB, dynamic scheduler node001-node006.
Compact JSON only; no checkpoint/trace writes or local native training.

## Run Receipt

Implementation/results commit a493a6f381; preregistration commit09825d9c15,
pushed before submission. Ten related focused tests passed. All four selected
Stage135 warm upper files remain available on the shared server filesystem.
Run: pointmaze_exploration_match_stage140_probe_20261008_r1.
t136990/root410037 and t136991/root410049 were queued at receipt time;
neither had an assigned node. Dynamic node001-node006, no aggregator.
Preregistration and compact dispatch receipt are retained in the run directory.

## Completed Result

t136990 finished on node005; t136991 on node001. Pulled297,302bytes of compact
JSON only. Exact budgets, registered seeds, paired means and mean-step RMS pass.
Total3,968episodes/4,761,600steps; no checkpoint or native trace pulled.

Equal-root Fisher-plus return increment over warm:

| Period | Mean std0.15 | Mean std0.05 | Sampled std0.15 | Sampled std0.05 |
| --- | ---: | ---: | ---: | ---: |
| 50 | +0.144867 | +0.156576 | +0.022887 | +0.067652 |
| 100 | -0.077936 | -0.067526 | +0.017315 | +0.032048 |

Reduced std improves the learning increment at all four root-periods in both
deployment modes, but mean still fails root410049/period100(-0.149377).
Warm sampled-minus-mean losses shrink from2.439557/4.115205 to2.152174/2.676937.
Even reduced-std Fisher remains below forecast in sampled deployment by
0.329550/1.954542 at50/100. Noise reduction helps, not a closed learning claim.
Next: mean-trajectory antithetic option queries versus replayed stochastic
credit, fixed step and fresh evaluation, with added query cost explicit.

## Limitations

Mean-output RMS matches on each training-state batch, not decoded physical control
distance or fresh-state KL. New label distributions differ with std. This is a
two-root development comparison, not a confirmation CI, joint HRL or an automatic
winner-selection rule. Earlier costs and negative results remain separate.
