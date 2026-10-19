# V125: consumed four-arm timing propagation

Read all 32 V123/V124 factorial paths, never replay or train. Reproduce original
actual arrival CV2, dispatch moments, executed holding and readiness first.
Reuse the completed-stop decomposition: release, holding, nonholding stop time
and remaining elapsed. Report all six contrasts with crossed actor/day intervals,
physical first timing/holding differences, vehicle-assignment changes and
actual-neighbour changes. No causal mediation claim from covariance accounting.

Spatial profiles follow each direction's real station order. Four-hour cohorts
are anchored to raw-bilateral actual launch by trip, retaining cross-boundary
incoming headways. Reconcile every spatial/cohort pooled-CV2 partition with its
global endpoint; do not average local CVs to obtain the global CV. Preserve
unavailable local metrics. Focus detailed returned spatial profiles on day
2231004 and both forecast-vs-raw-bilateral contrasts; full profiles stay remote.

No terminal-return timestamp exists in these records; last stop departure is not
substituted for one. Tests and analysis through scheduler on node001--node006,
independent CPU conda, four workers; only summary JSON/logs return locally.
No new day, trajectory, gain change, actor/critic/dual update or main switch.
