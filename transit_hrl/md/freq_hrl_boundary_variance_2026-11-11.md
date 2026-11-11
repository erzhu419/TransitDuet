# Cost-Accounted Boundary Variance Diagnostic

Keep the four C0 controllers/rules and the closed E8 decision unchanged. The
four boundary prechecks passed. Freeze eight NEW development prefixes/root,
three outcome-independent reachable checks/prefix, four window draws/check and
four paired suffix redraws/window. Prefixes exclude E8, other historical roles,
registered reservations and effective controller-training seeds.

Collect W, tail and G jointly. Report per-state within-window suffix variance,
corrected between-window variance, W/tail covariance and nested-mean variance.
Retain negative corrected estimates. Separately compare first/last two windows
and first/last two suffixes; include joint-zero counts in sign agreement. Keep
per-prefix summaries and roots separate. Across-fixed-state variance minus mean
sampling variance is descriptive heterogeneity, not proof of predictable signal.

Full cost/prefix is 103 complete 1,200-step episodes: one factual run plus three
checks each with two original arms and 32 nested arms. Total 3,955,200 steps;
eight unpinned 1CPU/2GiB jobs, four prefixes/494,400 steps each, on node001-node006.
No fitting, cut selection, controller update or policy-performance evaluation.
Raw boundaries/draws stay on the server; retrieve compact moments and costs only.

For J windows and K suffixes, the nested mean has variance B/J + S/(J*K).
The equal-pair-count independent-window projection is (B+S)/(J*K). It is an
analytical simulator-step comparison, not an additional sampled control or a
measured total-CPU comparison. With full replay charged, nesting itself is not
an efficiency claim: for nonnegative B it cannot beat independent draws for
estimating the unconditional mean at equal pair count.

Inspect both noise levels and their stability across development prefixes before
choosing a boundary-value training target. No critic fit, extra validation cohort,
budget extension or original confirmation follows automatically.

## Limitations

These are conditional draws given the simulator's realized latent driver state,
not all uncertainty from observable history. Four windows give noisy component
estimates. Suffix siblings, overlapping checks and two roots/task are not new
independent fitted policies. Corrected heterogeneity/sign agreement alone cannot
establish learnability, causal-state sufficiency or a performance gain.

## Release

Frozen implementation `a19d97c8e5`; 26 focused tests passed. Eight jobs
t214559-t214566 release only the declared 3,955,200-step variance diagnostic.
The [launch receipt](../results/plan_validity_boundary_variance_20261111/release.json)
records actual placement/progress separately from completed results. Original
C0 models/rules remain fixed and E8 stays outside this acquisition.
All eight jobs completed with exit 0 on node003-node006. The immutable launch
receipt retains the initial progress snapshot. No raw boundary data,
trajectories or models were downloaded.

## Completed Readout

Exact coverage: 32 prefixes, 96 states, 384 windows, 1,536 paired futures and
3,955,200 native steps; zero fits/optimizer updates. Acquisition used 3,425.594
CPU seconds; longest shard 468.915 seconds; peak process RSS 488.965 MiB.
The merged public-moment input is 153,051 transferred bytes. Fifteen focused
sampling/readout tests pass. See the [root/prefix summary](../results/plan_validity_boundary_variance_20261111/summary.json).

| Task/root | G suffix variance share | W window-half sign agreement | G window-half sign agreement | Nested/independent variance projection |
| --- | ---: | ---: | ---: | ---: |
| PointMaze/1972110000 | 75.2% | 22/24 | 16/24 | 1.745 |
| PointMaze/1972120000 | 71.7% | 21/24 | 19/24 | 1.848 |
| Reacher/1972210000 | 87.0% | 24/24 | 19/24 | 1.390 |
| Reacher/1972220000 | 76.7% | 24/24 | 12/24 | 1.700 |

No joint-zero states inflate those sign counts. G suffix-half agreement is
22/24, 21/24, 19/24 and 16/24 respectively. Three roots' mean G changes sign
across window halves; only PointMaze root2 retains its sign on both axes.
The largest Reacher root2 state contributes 72.3% of its estimated mean-label
variance. Its corrected across-state G heterogeneity is -0.003296, retained
as a noisy estimate, not evidence of negative variance or zero true signal.
Other roots have positive corrected heterogeneity, which is not predictability.

The suffix is the main uncertainty source; the W/tail covariance is retained.
Nesting did not buy unconditional-mean sampling efficiency at full replay cost.
Positive credit favors now over wait-one-check, not improvement over C0.

## Next Bounded Step

Use only the resident development cache for a boundary-tail learnability probe:
compare zero tail, an intervention-state predictor and a shared boundary-value
difference with matched effective capacity. Hold out entire prefixes, separate
suffix A fitting from suffix B readout, retain window halves and report each
root. These already inspected development prefixes are not fresh confirmation.
Boundary context is available at b, not deployable at the earlier decision t;
even a successful probe would still need a causal upstream estimator.
Freeze that probe before fitting. No new simulation, critic training, cut
selection or policy evaluation was submitted with this readout; HOLD remains.
