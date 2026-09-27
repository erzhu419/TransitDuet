# Stage-22 Frozen-Ridge Error Attribution

Retrospective diagnostic after Stage-21 failed. Freeze roots 209011/209061,
all 16 states/root, Stage-20 averaged-label folds, normalization and alpha=1.
Use Stage-18 eight-draw labels and Stage-21 64-draw labels already observed.
Preflight 208001 uses its two states and existing four-draw caches.
No model selection, new controller rollout, parameter update or deployment.

Reconstruct each fold's linear influence H from its saved design and penalty:
H = Q (D' W D + I)^(-1) D' W. Require H times original training means to
reproduce the saved out-of-path predictions. Training and query paths remain
disjoint. Score the frozen coefficients on old training means, independent
future means at training states, and independent means at held-out states.

Let p be the frozen prediction, m=H times fresh training means, y the fresh
query mean, v the estimated variance of m, and u that of y. Report:
- Training-label realization term: (p-m)^2-v.
- Signal-mapping mismatch term: (m-y)^2-v-u.
- Interaction: 2(p-m)(m-y)+2v.
Their sum is exactly the Stage-21 corrected error (p-y)^2-u. Preserve signed
estimates and interactions; do not turn them into additive error percentages.
Also report H-squared propagation of old training-label mean variances.

Equal-state/path summaries and per-path rows are descriptive. Signal mismatch
includes regularization, finite-state coverage and model representation; it
does not identify irreducible state insufficiency. Fresh labels are reused for
retrospective attribution, not a new independent test or a deployable refit.
No significance gate, tuning or extra sampling follows from this diagnostic.

Eight 40-dimensional influence solves/root, zero parameter updates and zero
new primitive steps. Use scheduleurm, one CPU/1.5 GB per root, dynamically on
node001-node006; return only a compact JSON, with no checkpoint or raw replay.

Implementation verification: 15 focused tests passed, including exact error
reconstruction, unbiased finite-noise corrections, negative-term preservation,
saved-coefficient replay, path isolation, input budgets and scheduler resources.

Preflight `t101429` passed on node004; full tasks `t101432/101433` completed
on node006/node005 at revision `8721fa2776`. The
[result](freq_hrl_stage22_ridge_error_result_2026-09-27.md) points to fixed-mapping
limitations rather than label precision as the next intervention. No new gate
or performance claim was introduced.
