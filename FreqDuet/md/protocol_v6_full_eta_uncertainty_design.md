# Full-ETA uncertainty after V158

Design only. V151/main,V154 moments and weights,V146(3s,0s),3600s window and
.1/.1 tails stay frozen. No uncertainty model is fitted or admitted here.

## Chosen Structure

Attach a state-conditional joint residual layer to the unchanged point forecast.
V151 remains the reference;the eight-day running version is still unpromoted.
Target the same physical follower/trip and target-ready clock as V158,not an
unmatched next departure. Use paired running/service/post-ready residual vectors
from the same archived prediction;post-ready keeps holding and execution wait
together when current-ready residence is inseparable. Preserve their dependence.
Opaque fallback points retain a separate full-residual stratum or abstention,
not an invented component split.
With e=prediction-truth,T=m-e;fixed-tail intervals would be
[m-Q_0.9(e|context),m-Q_0.1(e|context)]. No such quantiles are estimated here.

Prefer empirical residual blocks to an independent Gaussian sum or a constant
global covariance. Retain phase,route/stop,observable current service/source,
predicted remaining route and horizon information. V158 has not identified the
conditional tails or shown that within-cell dependence is irreducible noise.
Only query-time observables and frozen predicted clocks define conditions.

Retain day blocks with equal day mass and follower-target episode weighting;
do not make an all-day scalar bias the primary mechanism. Conditional day-by-
route/time variation may remain even when the whole-day mean effect is small.
Past daily means are not a prediction interval or a standard error of future
travel duration;within-day variation must remain in the predictive distribution.

## Next Implementation Boundary

First extract joint residual records from existing server archives,with query
time,target-ready/label-available time,observable context and support metadata.
Only labels strictly completed before a query can update its online residual
history. Keep unsupported/late-label states explicit;do not fill them with zeros.
Keep signed residual asymmetry around the fixed point rather than recentering
it through a new bias correction. The V158 cells lack per-query joint tails.

An eventual fixed interval protocol needs calibration and confirmation days
disjoint from prior fitting and one another. Previously consumed V145/V151 days
are development evidence,not fresh confirmation. Hold window/tails fixed;
assess full-ETA coverage,score,width,clearance/hour13 and support/identity strata.
V146 current-stage coverage does not certify a full-ETA interval. No controller
switch or replay is part of this design;prediction validity precedes any control test.

## Limitations

V158 separates global actor/day means,not every local day-by-context shift.
Its first/second moments cannot determine joint tails or predictive coverage.
Eight prior days mix latent day variation and finite-sample mean noise. This
design is not an interval result or evidence of passenger/control improvement.
