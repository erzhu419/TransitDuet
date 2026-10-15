# V50: completed-prefix conditioned travel

Freeze the real U200/L200 controller and V48 passenger_joint/holding models.
Use V47 completed full-boundary AVL legs with end_s strictly before each origin.
Previous-hour road means provide priors. Empirical-Bayes updates absorb completed
current-hour records, then the most recent 900-second subset; estimate pooling
variance across roads, and observation noise within completed hour groups.
Future unknown hourly regimes are not read or extrapolated from latent profiles.

Fit uncertainty from matured, strictly prefix-issued V50 travel innovations.
The initial origin uses completed historical within-hour deviations as its prior.
Positive empirical ratio marginals are recentered to arithmetic mean one. Compare
conditioned_mean, conditioned_independent and conditioned_shared; the latter uses
a Gaussian copula with a fresh shared road field every 300 seconds. Estimate its
shared fraction from repeated road/bin observations and retain independent noise.
This does not copy persistent three-hour mean drift into a future shared block.

Run smoke, 12-day preparation/prequential screening, then 612-origin frozen
closed-loop forecasts through scheduler on node001-node006. Prequential screen:
cohort MAE/coverage no worse than V46, no demand-cluster MAE regression >5%.
Closed-loop report raw/calibrated holding/passenger MAE, bias, CRPS, coverage and
event-clock errors versus V48 and historical; use demand clusters, not overlapping
windows, for inference. Preserve passenger mass and unchanged actual controls.
After travel validation, add turnover-conditioned seven-bin holding response as
a separate diagnostic, not a controller replacement. Retrieve summaries only.
