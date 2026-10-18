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
six new tests and eighteen consumed primitive paths. After syncing the real
smoke marker, review `t183619` launched on node006; both jobs use one CPU
and only node001--node006.
Only the smoke marker and final small summary synchronize, not timing records.

## Outcome And Decision

Review t183619 completed with exit 0, reading all 288 paths; all V108 supports,
moments, covariance collapses and crossed intervals reconstruct. Only the
1104546-byte final JSON returned. Bilateral025 minus preserve, arrival CV2:

| Contribution | Mean | Crossed actor/day 95% CI |
| --- | --- | --- |
| Nonholding stop allocation | +.002215 | [.001695,.002679] |
| Remaining elapsed allocation | +.005449 | [.003331,.007422] |
| Holding/stop cross | +.003746 | [.002351,.005159] |
| Holding/remaining cross | +.011457 | [.005480,.017314] |

Both allocations are positive in both actor means; stop in 16/16 day means,
remaining in 14/16. Remaining-elapsed diagonal alone is inconclusive:
-.000792 [-.003926,.002155]; stop diagonal +.000591 [.000334,.000847].
At forward-direction stop 20, remaining diagonal actually falls by .012164
while holding/remaining cross rises by .040628; stop allocation +.004192,
remaining +.005804, raw headway variance +1292 s2. Nearly all CV2 change
is on unchanged predecessor edges. Thus this is not simply more independent
travel noise or overtaking. Departures show the same qualitative split.

Keep this as two-actor covariance evidence, not causal mediation or a new
controller result. The next hypothesis is V110: fixed-gain bilateral feedback
plus causal, observed nonholding-disturbance feedforward, tested once on fresh
development days. This tests compensation rather than fitting another absolute
travel predictor. No V107 gate/sample extension or main-policy switch.
