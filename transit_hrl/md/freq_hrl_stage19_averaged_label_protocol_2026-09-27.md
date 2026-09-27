# Stage-19 Cached-Label Averaging Diagnostic

Implementation `dc8a2bf316`; 22 focused tests passed, including default critic
behavior, shared normalization and exclusion of all held-out labels/costs.

Freeze before fitting: roots 209011/209061, operational preflight 208001.
Use exactly Stage-18's 16 selected opportunities/root (two per path) and their
eight conditional-future contrasts, joined to Stage-16 compact endpoints.
Compare replica index 0 against the mean of all eight as the training label.
No label-based selection, additional replay, controller training or new roots.

Both critics use the Stage-18 compact layout (40 features plus 384 zero slots),
two width-64 hidden layers, 31,425 parameters, paired-difference loss, 64 Adam
updates at 0.001 and seed (root, held-out path, 16016). Train-only feature
normalization, baseline rate and target scale use the SAME original endpoints
in both treatments; only contrast labels change. Hold out an entire path,
including every future replica; 14 training pairs and two query pairs/fold.
The single label is contained in the eight-label mean. Training state counts
and optimizer budgets match; label simulation budgets are one versus eight,
not equal, although all labels are cached and no new environment cost is incurred.

Score both out-of-path predictions against each query's eight-future mean;
subtract mean(sample variance / 8) from MSE and retain negative estimates.
Qualification requires averaged labels to beat single labels AND zero on BOTH
roots. The correction is common to all methods and cannot alter their ranking.
Report raw/corrected MSE and per-path results, without retuning on either metric.
Sixteen critic fits/1,024 updates per root; zero new primitive steps. Inherited
Stage-18 replay cost remains 547,200 steps/root plus earlier training/supervision.
Preflight uses its existing two paths, one opportunity/path, four replicas:
four fits/256 updates. Schedule through node001-node006 without node pins.

This is a small, reused-development-state representation/label diagnostic,
not independent confirmation or a deployed policy. It tests whether averaging
improves this learner, not whether future noise is the sole cause of failure.
No deployment or root expansion follows automatically, even on qualification.

Preflight `t101351` completed on node006. Its 6,452-byte JSON passed paired
normalization, path isolation and zero-new-step accounting checks: two pairs,
eight cached labels, four fits/256 updates. Both treatments predicted zero;
raw MSE was 2.549147e-8 for both and zero, corrected MSE -4.954739e-8.
Qualification failed on a tie. No settings were changed before development.
