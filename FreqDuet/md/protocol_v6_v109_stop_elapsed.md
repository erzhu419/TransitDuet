# V109: nonholding-stop versus remaining-elapsed accounting

Reuse all 288 V108 primitive records on the servers; no simulator or checkpoint
loads. Keep the same two source-order actors, sixteen consumed days, five frozen
controllers and sampler averaging. Extend the existing V92 four-component
accounting: release, executed holding, observed nonholding stop time, remaining
elapsed. Stop time is departure minus arrival minus executed holding, accumulated
only after that stop; remaining elapsed includes movement and any unobserved
terminal/clock time, not a pure driving-time label.

Reconstruct every V108 full/stop/direction/four-hour support and moment, and
collapse four-component covariance back to its three-component parent. Retain
negative cross covariances, actual predecessor changes, full-day service-boundary
contributions and clock-band sample counts. Global effects use the existing
crossed actor/day bootstrap after sampler averaging; local scopes are descriptive.

Smoke: V92 tests, new scoped-accounting tests and the first consumed day for
each actor (18 existing path reads). Formal review waits for the smoke marker.
Everything runs through scheduler on node001--node006 in the isolated CPU
environment. Only markers and the final small JSON return locally.

Decision: use the split of holding/other covariance and downstream raw variance
to decide whether a subsequent control candidate needs stop-service/load
feedback, remaining-time propagation, or both. No new gate or candidate is
selected from this audit. V107's confirmation budget stays closed; actors,
critics, duals, main policy and all original thresholds remain frozen.

## Run Record

Frozen source `ac336aaecd`. Smoke `t183618` passed on node006: five V92 tests,
six new tests and eighteen consumed primitive paths. Review `t183619` waits
for that real smoke marker; both jobs use one CPU and only node001--node006.
Only the smoke marker and final small summary synchronize, not timing records.
