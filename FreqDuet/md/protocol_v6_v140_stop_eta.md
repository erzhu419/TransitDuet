# V140: passive stop-phase ETA

One fixed shadow extension on the same12 V137 paths; no replay, new demand,
action selection or actor/critic/dual update. Tests and audit run on server via
scheduler. The existing moving, launch-handoff and raw fallback logic stays intact.

For strictly observed pending stops: condition completed service durations on
surviving the current elapsed service age; then add completed-prefix expected
holding/ticks and the original downstream component ETA. After ready and an
earlier command are observed, use the known command execution finish clock.
Use the original1h window/all-past fallback; unsupported cases retain native ETA.
Current/future ready, departure and command labels are excluded until observed.

Report fixed-support native-fallback MAE and matched source-switch gap-change
error, by path/phase as well as pooled. Unsupported counts and current-service
interval coverage remain explicit. No formal policy promotion from these data.

Outcome: source4df22f4fa6, t195272/node006 exit0; 8 tests pass, known-command
finish clocks match all saved commands. Only40790-byte summary fetched.
Fixed fallback support11676: MAE66.548->46.007s, bias15.692->0.468s; all12
paths improve. Extension applies10943 times; all remaining cases retain native
ETA in that fixed-support metric.721 source-switch pairs: gap-change error
76.831->15.752s; all6 pairs improve, including unsupported retained predictions.

Limitations: outcome-selected consumed development paths; future route holding
is still a completed-history mean. Current-service nominal80% interval coverage
is only61-67% by path.733 fallbacks remain native:492 ready without an earlier
observable command,199 without a service survivor,42 position/event boundaries.
This is prediction evidence, not control gain.

Next: keep V140 fixed as the predictor reference. Address the not-yet-observed
command branch and diagnose interval undercoverage before independent passive
confirmation or any control trial. No window/quantile tuning on these days;
holding gain, action bins and the main policy remain unchanged.
