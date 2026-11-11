# Cache-Only Boundary Tail Probe

Four root-specific jobs reuse the completed 4x4 boundary banks: eight development
prefixes, 24 states and 96 paired boundary windows/root. No simulation, controller
loading, policy update, cut search or E8 reuse. These exposed prefixes are not a
fresh confirmation cohort. Freeze this protocol before reading fitted results.

Fit tail = C0_suffix(wait) - C0_suffix(now), using suffix indices 0/1 only.
Indices 2/3 remain independent conditional suffix readout on the SAME windows.
Compare zero, A-only constant, causal state at t, and V(boundary_wait)-V(boundary_now).
Use the existing 39 causal interactions plus phase/remaining/call-spent, and
39 interactions each with remaining and call-spent: 120 columns in both views.
Every t is an available C0 check, so its current-bin call is unspent.

Fix total effective DF at 3: causal t has one intercept plus two slope DF;
boundary difference has three slope DF and no intercept. Sign augmentation
enforces a shared potential, zero on identical states and reversal antisymmetry;
the reversed rows are not extra observations. No hyperparameter selection.
Each root fits the full A bank plus eight whole-prefix leave-one-out folds:
18 ridge fits and nine constant fits, 72/36 across all four roots.

Report training A, training B, prefix-OOF A/B, B window halves and each prefix.
Subtract conditional suffix-mean variance only for independent evaluations;
also report OOF A/B error cross-products and retain negative estimates. No CI
from sibling suffixes, windows, overlapping checks or overlapping fitted folds.
Boundary delta qualifies only if OOF B MSE beats zero, constant and causal t
overall AND in each window half, on every root; qualification is descriptive.

## Limits And Next Decision

Realized boundary information is label-side and unavailable at t; a successful
probe would still require an upstream causal estimator. A-only training error
does not establish transfer. Conditional mean accuracy does not establish a
policy or complete-episode gain. If the held-prefix boundary probe fails, do
not distill or deploy it. No follow-up is automatic; `PILOT_RECIPE_HOLD` remains.
Raw rows, coefficients and window predictions stay on the server; fetch only
compact public losses and measured costs. Four unpinned 1CPU/2GiB jobs use
node001-node006 through scheduler, not Slurm.

## Result

Frozen code `116f065e25`; 14 focused tests passed. Jobs t214574-t214577 completed
their public workers on node004/node006: exactly 72 ridge/36 constant fits,
zero native steps. Measured phases used 2.123 CPU seconds, excluding process
imports and scheduler startup. Public results: 92,738 bytes; derived fixed-model
attribution: 3,073 bytes. No raw bank or coefficient downloads.

Prefix-OOF suffix-B tail MSE, lower is better:

| Task/root | Zero | Causal t | Boundary delta | Frozen rule passes |
| --- | ---: | ---: | ---: | --- |
| PointMaze/1972110000 | 0.001948 | 0.001995 | 160.668414 | No |
| PointMaze/1972120000 | 0.002220 | 0.002259 | 5631.627653 | No |
| Reacher/1972210000 | 0.026331 | 0.028135 | 0.025025 | Yes |
| Reacher/1972220000 | 0.073664 | 0.077064 | 0.073133 | No |

Reacher root1 beats zero by 4.96% and causal t by 11.05%, with both B window
halves improving. Root2's overall 0.72% improvement over zero does not survive
the halves: MSE-minus-zero is +0.004263/-0.005325. Its training-B error already
worsens, so this is not solely a new-prefix transfer problem. Causal t loses
to zero on all four OOF B roots. The cross-root qualifying rule fails.

PointMaze's training-B boundary errors improve, but OOF explodes on prefixes
1972175003/1972185005. Fixed-coefficient attribution identifies the dominant
`velocity_001_minus_050_norm__spent` term: training contrast scales 6.25e-7 and
6.66e-8, held contrasts 0.576/0.900, contributions 124.22/734.60 versus B tails
0.0168/0.0565. This is reachable normalization/support failure, not evidence
that tail has no signal. The ungated parent has healthy training-arm standard
deviation 0.557/0.788; gating/differencing produced the near-zero scale.

Next prepare a single geometry repair: scale parent state features on training
prefixes before clock/resource products and arm subtraction, with no subsequent
contrast-variance restandardization. Keep total DF, A/B split, folds, labels and
baselines fixed; do not clip predictions using held outcomes. These failed v1
results remain intact. No extra fit budget or policy trial has been released.

Sources: [public losses/costs](../results/plan_validity_boundary_tail_20261111/summary.json),
[term attribution](../results/plan_validity_boundary_tail_20261111/extrapolation_attribution.json),
[execution receipt](../results/plan_validity_boundary_tail_20261111/release.json).
