# jtl110cpu Stale-Task Quarantine Record

Date: 2026-08-08; reconciled: 2026-08-10

## 2026-08-10 Reconciliation

The original 81-record blanket quarantine below is superseded. A record-level
audit found completion summaries for 75 records; those records do not require
process cancellation and must be adjudicated from their artifacts rather than
from the stale scheduler label. Six records still have no declared result
directory or synchronized completion artifact:

- `t66714`
- `t66715`
- `t66718`
- `t66720`
- `t66721`
- `t66724`

All six are obsolete `nested_hpo_v6_pilot_20260803` flat-GRU HPO cells and
remain excluded from every evidence and dependency count. On 2026-08-10 an
exact forced cancellation selected all six, but scheduler reported
`kill_failures=6` because the configured `jtl110cpu` endpoint closed the SSH
connection. The records therefore remain `running`; this status does not prove
that their remote PIDs are alive. Do not relabel them `done` or `cancelled`
until process-tree termination or a valid completion artifact is observed.

## Observed State

At the time of the original incident, the scheduler contained 81 records with
signature prefix
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

The six unresolved records identified by the 2026-08-10 reconciliation are
excluded from:

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
