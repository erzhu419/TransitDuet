# V49: travel and holding residual dependence

Freeze the U200/L200 controller and reuse the 12 certified V45/V47 service-day
records. Six previously examined demand seeds are the inference clusters;
average the two actor seeds within each demand before descriptive bootstrap CIs.
All computation runs through scheduler on node001-node006. Retrieve summaries only.

For each eligible holding decision and completed travel leg starting in
(3600,49500], use the latest stored V45/V46 model issued strictly before that
decision/departure. No future fit or overlapping-window duplication. Join teacher
holding features to exact executed clock/direction/stop identities; reconstruct
full terminal-inclusive legs with bus/trip identity. Keep unsupported V46
terminal-leg fallbacks visible. Future true features are residual scoring labels.

Audit raw and retrospectively day/direction/stop/hour-centered residuals. Report
same-bus within/across-trip event lags 1-3, same-directed-stop different-bus lags
1-3, and all different-bus pairs at 0-300/300-900/900-1800 seconds, separately
within/across direction. Compare V46 shared-block implied covariance on pairs
with the same issued origin; holding innovations are independently sampled
conditional on factual features. Compare incoming completed travel residual to
the same vehicle's next holding residual. Centering is not a deployable model.

Nonoverlapping 900/1800-second decision/departure windows partition squared
residual sums exactly into diagonal, same-bus, same-stop-other-bus, other same-
direction and opposite-direction terms; also report lagged window sums. These
are innovation sums, not clipped holding exposure or actual arrival-clock error.
Separate mean misspecification from dependence; correlations alone do not prove
causal propagation. Smoke tests cover identity, strict issue time, covariance
arithmetic and independent/correlated controls before the 12-day audit. No policy
promotion or new counterfactual superiority claim from this diagnostic.

Centering control: 200 fixed-seed within-day/direction/stop/hour label permutations
retain stratum distributions and sampling geometry. Compare lag-one event pairs,
nonoverlapping window-sum correlations/MSE, and incoming travel/holding residuals
with this null. Recompute centering on the preserved strata. Null quantiles are
shuffle reference ranges, not population CIs; demand-cluster bootstrap summarizes
observed-minus-null effects. This controls mechanically induced negative
dependence, particularly the two half-hour windows sharing an hourly mean.

## Results And Next Step
All 12 days completed: matrix t155813-t155817, aggregate t155818; 200-shuffle
controls t155831-t155835, aggregate t155837. Smoke t155810 passed 11 tests;
t155836 passed 12. Earlier smoke attempts exposed two test-assertion issues
(pandas method/column collision and floating equality), repaired without
changing experiment parameters; failed/cancelled attempts remain in scheduler.
Audit source 3e27ae6061, permutation code 070061a960, test-only repair 8a9068761a.

Raw same-leg/next-vehicle travel correlation is 0.516. Its excess over the
within-hour/stop permutation null is +0.060 [0.052,0.068]. Holding excess is
+0.170 [0.112,0.229] across trips of one bus, but -0.180 [-0.187,-0.173]
between adjacent vehicles at a stop. Incoming travel-to-holding excess is
-0.069 [-0.072,-0.067]. Adjacent 30-minute window correlations do not show extra
positive persistence beyond the null (travel -0.020; holding -0.004).
Raw 30-minute travel/holding innovation-sum MSE divided by independent model
variance is 9.48/1.03. These are conditional factual-state diagnostics.

Next: prioritize completed-prefix, road/time-conditioned travel means and
short shared residual structure, then turnover-conditioned holding response.
Validate each causally and in closed-loop forecasting before any controller
change. A global positive holding-noise process is not supported by this audit.
