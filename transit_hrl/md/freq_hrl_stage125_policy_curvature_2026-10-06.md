# Stage125: Native Policy-Curvature Step

Stage124 preserves a useful direction but its full-policy step does not
consistently improve native return. Learn step size from the actual native
episode objective instead of copying the single-option Fisher radius.

Reuse each root/period's frozen Stage124 direction. On16 fresh training
scenarios and two lower-noise panels, evaluate full-policy steps0,+1,-1.
Fit q(t)=a*t+b*t^2 from paired return differences and maximize it over[0,1].
Fit A/B separately for crossfit tests and pool both for the deployed step.
No evaluation rewards enter fitting; no direction or checkpoint is selected
by crossfit rewards. The inherited direction and maximum step stay fixed.

Evaluate32 separate new1200-step scenarios at periods50/100. Compare scaled
ascent against flat, forecast, unscaled ascent, scaled descent and blinded.
Lower/critics/variance, the0.05 response limit and0.5 practical-gain threshold
remain unchanged. Native episode budgets per root:192training+128crossfit+
384evaluation=704episodes/844,800steps. Inherited Stage123/124 costs are
reported separately. Two tasks, five CPUs/8GiB each, dynamic node001-node006.
Only compact JSON is fetched; two final upper weights per root stay server-side.

Ten tests passed: quadratic maximizer, weight/variance preservation,
training/evaluation separation, native common-noise/blinded/freeze checks,
source direction reuse, measured episode budgets and fresh seed rosters.

## Limitations

Two-root development, not joint-HRL or independent confirmation. The native
return quadratic is approximate; crossfit and fresh deployment must check it.
A zero fitted step is recorded as inactive upper, not a successful learned
plan. Stage124 failure and earlier HOLD/negative gates remain unchanged.

## Run Receipt

Code `404191251d`; run `pointmaze_policy_curvature_stage125_pilot_20261006_r1`.
`t135907` root410011 and `t135908` root410023 accepted, initially queued.
Fixed commands, training/evaluation rosters, budgets and fitting objective are
saved in `preregistration.json`. No Stage124 seed extension submitted.

## Completed Result

Both tasks DONE (node004/node006); fetched35.6KB JSON only. Pooled scales
for root410011 p50/p100 are1/1; root410023 scales are0.1473/0.9590.
Fresh ascent-minus-forecast gains: -0.09993/-0.03277 and+0.04230/+0.21833.
Equal-root means=-0.02882/+0.09278. Shrinking root410023 p50 improves over
its unit step by+0.23621, but its B-to-A training crossfit remains-0.04714.
No practical gain claim closes. Root410011's positive training gains do not
transfer. Next: matched empirical-Fisher versus Euclidean direction learning,
with independent new calibration/evaluation, not another scalar-only adjustment.
