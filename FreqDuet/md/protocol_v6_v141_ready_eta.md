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

Outcome: source9adf0bc09c, t195273/node006 exit0; 8 tests pass; only37493-byte
summary fetched. All492 ready cases receive the extension: MAE63.305->45.029s.
Full11676 fallback support: V14046.007->V14145.237s.721 source-switch pairs:
15.752->11.817s; all6 improve.11/12 individual paths improve;933059/2351002/old
worsens0.290s overall (new-branch support36.241->43.042s). Retain V141 as a
candidate, not a universal replacement for V140.

Frozen service intervals:7128 queries/7126 episodes, coverage63.538% vs
episode-weighted63.528%; duplicate weighting is negligible. Median support9;
singletons(n230) have zero width and22.17% coverage; support2-4(n881)39.50%,
support10-19(n1954)72.82%. Past-sample containment79.04% is not future coverage.
Lower/upper misses18.48%/17.99%, mean bias+0.077s: a common offset is not the
primary repair. Small-support construction is implicated, not established as
the sole cause; do not infer distribution drift merely from support escapes.

Limitations: consumed outcome-selected paths, no independent policy evidence;
past-sample containment is not predictive calibration.

Next: keep V140/V141 point predictors fixed. Address finite-sample interval
construction/support sufficiency before independent passive confirmation;
do not tune windows/quantiles or switch the controller on these six pairs.
