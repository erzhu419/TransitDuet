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

## Limits And Next Step

Fourteen focused tests passed, including coordinate invariance, allocation
closure and zero-fit accounting. Geometry/support associations do not identify
the causal effect of normalization; no alternative-geometry fit is released.
Two-label noise propagation is descriptive, not a CI or correction to the gap.
Use this diagnosis to choose the next bounded training-A question; E remains
fixed reporting evidence and original policy confirmation stays on HOLD.
