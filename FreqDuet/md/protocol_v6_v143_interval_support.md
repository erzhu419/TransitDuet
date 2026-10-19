# V143: read-only support and hour-13 diagnosis

Copy V142 bounds and V140/V141 point forecasts without regeneration. Read only
the same twelve V137 paths, no replay or training. Explain support loss as
no completed history, fewer than nine total prefix labels, a sparse one-hour
window, or elapsed-duration survivor depletion. Preserve the original fallback
rule: no survivors does not trigger an expanded window.

For hour13, report paths, stations, directions and lower/upper misses, observed
duration versus completed historical durations, and pending-label selection.
Pending final durations are retrospective diagnostics, never predictor inputs.
Reproduce V142 coverage and bounds before interpreting the decomposition.
No tuning on these consumed selected paths; main policy stays frozen.

Results: source46f879f9ae, scheduler t195285/node006 exit0;13 tests pass.
V142 bounds and coverage reproduced exactly. Only the60360-byte summary was
fetched; all detailed trajectories and diagnostic rows remain on the server.

Of3353 unsupported service requests,399 have fewer than nine completed prefix
labels,471 have a sparse one-hour pool,2483 lose support through elapsed-time
conditioning. In the last group mean pool9.43 becomes4.97 survivors. No case in
this already selected cohort lacks completed history altogether. Ready's56
unsupported requests split28 prefix/28 window scarcity, not survivor depletion.

Hour13 has122 requests,98 finite,74 covered,19 lower misses and5 upper misses.
The19 lower misses have mean actual total service6.21s versus past14.76s;
mean pool and survivor counts both10, so age filtering removed no observations
in those cases. Mean miss distance below the bound3.68s. Lower misses occur on
ten paths, not one exceptional path. Both modes are below80% (old76.36%,
handoff74.42%). Pending events average one per query, not a large label backlog.

Limitations: these are consumed outcome-selected, resolved stage-duration cases,
not a complete eligibility cohort. Historical/current duration mismatch is
observed; demand decline or a causal censoring mechanism is not established.
The98 finite queries are not98 independent days. No window/tail adjustment.

Next: retain explicit abstention for low support; use the separate fixed-model
independent validation design. It requires unused days and complete eligibility
logging, not retrospective threshold rescue. No new path collection is launched.
