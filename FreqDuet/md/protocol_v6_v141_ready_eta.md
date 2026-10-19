# V141: passive unissued-ready ETA

Keep V140 as a frozen reference. Extend only ready-without-prior-command cases
using strictly completed ready-to-departure durations, conditional on elapsed
ready age. Current future commands are not inputs. Keep the original1h window,
historical fallback, p10/p90 and all other V140 forecasts unchanged.

On the same12 saved paths, compare full native-fallback support and all721
source-switch gap changes against both V140 and native ETA. No replay, action
selection, training or main-policy change. Run tests/audit through scheduler;
full details remain remote, only a small final summary is fetched.

Diagnose frozen V140 service intervals by lower/upper misses, past empirical
containment, support count, hour and direction. Compare query weighting,
episode weighting and first-query coverage. These groups are diagnostic, not
new gates or tuned quantiles; missing intervals remain explicitly missing.

Limitations: consumed outcome-selected paths, no independent policy evidence;
past-sample containment is not predictive calibration.
