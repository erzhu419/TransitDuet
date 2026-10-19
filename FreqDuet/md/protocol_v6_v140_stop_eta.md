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

Limitations: outcome-selected consumed development paths; future route holding
is still a completed-history mean. This is prediction evidence, not control gain.
