# jtl110cpu Stale-Task Quarantine Record

Date: 2026-08-08

## Observed State

The scheduler contains 81 records with signature prefix
`Freq-HRL/hpo-v1/nested_hpo_v6_pilot_20260803/` and node `jtl110cpu` that still
display `running` after more than 110 hours.

These records are not accepted as evidence of live computation:

- a forced batch cancellation was requested;
- scheduler metadata reports `cancel requested but remote process-tree
  termination was not confirmed`;
- the last kill attempt failed during SSH key exchange with
  `kex_exchange_identification` / connection closure on the configured remote
  endpoint;
- there are no newly synchronized completion artifacts establishing that the
  remote PIDs are alive or that the cells finished validly.

The apparent 100% CPU and fixed RAM fields are stale scheduler observations,
not current remote telemetry. They must not be interpreted as live utilization.

## Quarantine Rule

All 81 records are excluded from:

- experiment completion counts;
- result merging;
- dependency barriers;
- claim matrices and manuscript statistics;
- retry or tuning decisions.

They remain recorded rather than being relabeled `done`, because remote
process-tree termination could not be verified. If the endpoint becomes
reachable, reconcile each recorded PID and result directory before either
cancelling or archiving the records.

All new Freq-HRL CPU work is restricted to dynamic placement across
`node001` through `node006`.
