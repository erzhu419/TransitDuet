# Stage-15 One-Check Advantage Development Protocol

Implementation revision: `451b18e29a`.

Frozen roots: 209011/209061; preflight: 208001. Reuse the Stage-12 controller,
reference trigger, and seed roles. Stage-13/14 evaluation outcomes are never
fitting labels. This is one approximate policy-improvement step, not another
threshold sweep or a new independent confirmation.

On each of eight branch-fit paths, roll out the frozen Stage-12 trigger.
Using RNG key (root, path, 15015), choose 12 distinct noninitial/nonfinal bins
and one uniformly drawn eligible check at or before the factual call, on the
5-step grid strictly before offset 25. Replay now versus skipping that check;
both arms then use the same Stage-12 trigger on their own observations.
Exact prefixes, factual episode replay, and one call per bin are required.

Fit full-episode ISE(wait) minus ISE(now) using the existing 39 causal
interaction features plus within-bin offset and remaining-episode fractions.
Use the existing Ridge alpha grid and leave-one-path-out MSE selection on
branch-fit paths only. Deploy once, choosing now at nonnegative predicted
advantage; offset 25 still forces the single budgeted call. There is no
quantile calibration, additional policy iteration, or evaluation-time fitting.

Additional fitting costs 9,600 reference-trajectory steps and 230,400 paired
steps/root; held-out reference and candidate evaluations each cost 19,200.
These are additional to inherited Stage-12 supervision and replayed controller
training, not an equal-total-training-cost claim. Preflight uses two fit
paths, two pairs/path, and the existing 300-step/two-iteration controller.

Evaluate all 16 frozen held-out paths/root. Advance only if mean candidate
episode ISE beats fixed, Stage-12, and Stage-9 on BOTH roots. Otherwise retain
the failure without threshold tuning, extra iterations, or sequential roots.
Use scheduleurm node001-node006 (one CPU, 1536 MB/cell); sync compact JSON only.
