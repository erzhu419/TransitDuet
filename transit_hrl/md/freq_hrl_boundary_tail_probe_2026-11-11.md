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
