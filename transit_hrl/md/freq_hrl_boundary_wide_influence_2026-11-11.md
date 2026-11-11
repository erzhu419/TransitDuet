# Reacher1 Fixed-Fold Influence And Geometry

Reuse Reacher1's 32 Wide training-A prefixes and all 64 saved reciprocal-view
folds. No new fit, native step, B/E target, model selection or source-model change.
One 1CPU/2GiB scheduler task on node001-node006; fetch only public summaries.

Convert each saved parent-scaled potential to the common raw 120-column basis,
including centering-induced clock terms. Reproduce held predictions and matched
individual-model moments, then allocate the original loss gap over four feature
blocks without replacing the matched control by an ensemble prediction.

With each stored alpha/geometry fixed, derive its linear label response. Allocate
held-versus-zero loss over donor prefixes, and matched-versus-zero loss over own
and other prefix labels. These are exact additive quadratic-loss allocations,
not effects of deleting/refitting a prefix. Reuse the shared geometry of both A
views: 32 SVDs, 24,576 saved-fold predictions and 384 frozen-source checks.
Report all prefixes' ridge leverage, null-space share, scale/mean shifts relative
to full Wide geometry, and descriptive two-A noise propagation. No new parameters.

## Completed Result

t214741 completed on node004: exact 32 geometry decompositions, 24,576 fold
scores and 384 source checks; zero fits/native steps, 0.605 measured CPU seconds.
Only 43,512 public bytes returned. [Result](../results/plan_validity_boundary_wide_influence_20261111/summary.json),
[derived diagnosis](../results/plan_validity_boundary_wide_influence_20261111/diagnosis.json)
and [receipt](../results/plan_validity_boundary_wide_influence_20261111/release.json) retain provenance.

Prefix 1972277014 contributes 72.88% of the net 0.003574 prefix gap, positive
in both A views (0.07865/0.08806). The three largest gaps contribute 94.27%.
Omitting the largest readout prefix without refitting leaves +0.001001.
The largest-gap prefix has 16.96x training-mean ridge leverage, but negligible
null-space share. Across folds, parent-scale ratios are 0.870-1.016 and maximum
standardized mean shift is 0.02465: no recurrence of near-zero scale explosion.

In common raw coordinates, state terms allocate 82.81% of the gap. Squared
prediction terms contribute -0.000159, alignment +0.003733; the loss is not
explained by increasing prediction magnitude. Matched-future gain decomposes
into own-prefix label allocation -0.002454 plus other-prefix +0.000680, in
agreement across A views. This locates reliance on local prefix information,
without identifying the effect of deleting that prefix or changing geometry.

## Limits And Next Step

Fourteen focused tests passed, including coordinate invariance, allocation
closure and zero-fit accounting. Geometry/support associations do not identify
the causal effect of normalization; no alternative-geometry fit is released.
Two-label noise propagation is descriptive, not a CI or correction to the gap.
Use this diagnosis to choose the next bounded training-A question; E remains
fixed reporting evidence and original policy confirmation stays on HOLD.
Prioritize transferable conditioning/coverage of influential high-leverage
states at fixed capacity, rather than repeat a near-zero normalization repair.
Any such training comparison needs its own frozen A-only design; no further
fit, native budget or E-based tuning is released by this attribution.
