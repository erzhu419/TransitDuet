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

## Outcome

Source a112b40ad0, scheduler t195167 on node006: four tests, all 32 original
endpoints and exact covariance/profile partitions pass; actual child exit 0.
Only 617,478-byte summary returned. Day 2231004 split bilateral versus raw
bilateral: launch CV2 +0.001816, arrival +0.003968 (variance +507.95 s2,
normalization only +0.000133). Split funded: launch +0.005022, arrival +0.006294.
Both actors already worsen forward-direction launch/first-stop regularity,
followed by amplification around forward stop 6. No vehicle assignment or
arrival-predecessor changes. Holding differs before first launch differs;
actual launch changes therefore do not establish an independent upper cause.
Next V126 splits saved actual launch into first eligibility and readiness lag;
no rule change or replay is needed for that accounting.
