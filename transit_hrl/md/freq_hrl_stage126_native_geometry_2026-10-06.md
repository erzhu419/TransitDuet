# Stage126: Native Upper Mean Geometry

Stage125 fixes one excessive step but leaves training-to-deployment transfer
unresolved. Test whether Euclidean pooling over392 correlated state channels
is a bottleneck. Keep the linear Gaussian upper, its observations, native
Stage123 labels, decoder, lower teacher, critics and variance unchanged.

Reconstruct eight cached causal states per root/period. Center and standardize
their nonconstant columns, divide by sqrt(active dimensions), and retain a
bias column. Solve the native mean gradient in this empirical-Fisher coordinate
system with fixed unit damping, then map the weights back to the original
state space. Deployment still uses the same actor forward. Match Stage124's
fixed Fisher radius0.000555556; no label or direction filtering.

On16 fresh scenarios/two noise panels, both directions receive the same full-
policy0/+1/-1 calibration, pooled quadratic step fitting and A/B crossfit.
Evaluate32 separate new scenarios at both periods50/100. Compare natural
against Euclidean, forecast, flat, opposite-sign and blinded controls.
The0.05 response limit and0.5 practical-gain threshold remain fixed.

Per root:16 state replays+320training+192crossfit+384evaluation=
912episodes/1,094,400steps. Stage123/124 costs are reported separately.
Four workers+parent/8GiB, dynamic scheduler node001-node006. Only compact
JSON returns locally; two final upper weights per root stay server-side.
Four tests passed: dual/primal solve, coordinate-unit invariance, matched KL
and frozen source, full paired runner budget and training-only fitting.

## Limitations

This changes optimization geometry, not available information. Sparse cached
labels and two development roots still limit generalization evidence. The
natural direction may fail; a zero learned step is inactive, not upper gain.
No earlier HOLD/negative result, promotion or joint-HRL claim is changed.

## Run Receipt

Code `c891f819d2`; run `pointmaze_native_geometry_stage126_pilot_20261006_r1`.
`t135928` root410011 and `t135929` root410023 accepted, initially queued.
Fixed commands, normalization/damping, matched calibration, fresh rosters and
budgets are saved in `preregistration.json`. No Stage125 seed extension submitted.

## Completed Result

Both tasks finished with the registered budget:1,824episodes/2,188,800steps.
Root410011 natural-minus-forecast is+0.123029(period50)/+0.069837(period100),
and natural-minus-Euclidean is+0.083293/+0.069837. Root410023 selects scale0
at both periods: learned action and response are zero, so this is inactive,
not a successful upper policy. Equal-root effects are+0.061515/+0.034918,
below the unchanged0.5 practical-gain threshold.

All32 cached-query natural first-order alignments are positive, but root410023
has negative full-policy training slopes. Root410011 also has negative B-to-A
crossfit gains. Geometry improves local alignment, not reliable deployment.
Next test: fresh decision-complete native credit versus sparse-time credit,
with identical actor, Fisher geometry, authority and training-only calibration.
