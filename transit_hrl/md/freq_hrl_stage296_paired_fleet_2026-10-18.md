# Stage296: Independent Paired Fleet-Risk Calibration

Run: `native_transit_paired_fleet_stage296_20261018_r1`.

Use new scenes: 80 fit prefixes, 60 independent calibration prefixes, 120
untouched test prefixes, and ten separate software pilots. Balance both roots,
all five regimes and six probes. Stage294 E32 is excluded from fitting,
calibration, model/threshold choice and evaluation. Existing raw/247 models
and lower weights are reused, not retrained.

Keep the P8 wait-constrained proposal and S8 zero-event abstention unchanged.
Learn only paired fleet delta versus each profile's own raw action, using
independent T16 labels, preaction occupancy/timetable features and P8/S8
summary statistics. One 26-feature ridge1, no hyperparameter grid. Save the
fit model before opening calibration T16. Independently estimate selected
population residual bias and stratified prefix-cluster SE; use upward
adjustment max(0, bias+1.64485 SE). This is score calibration, not an actionwise
risk certificate. A same-recipe S8 bias-only control isolates regression value.

The risk score can only retain the original service proposal or fall back to
raw. No service refit or action reranking. Freeze calibrated model, all test
actions and expected count-matched thinning of baseline abstention changes
before test E32. All states remain in the denominator, including exact zeros.
P/S/T/E draws are 0-7 / 8-15 / 16-31 / 32-63, with disjoint scene-role seeds.

Primary: cost and wait versus raw must pass both prefix-bootstrap and shared
future MC 95% intervals and improve in both E16 halves. Service cost and wait
must also improve, retaining at least 50% of baseline abstention waiting gain
in pooled E32 and both halves. All-raw fallback cannot pass. Fleet improvement
versus abstention and cost advantage versus matched thinning are separate tests.
Average fixed profiles before 120-prefix inference; profiles are not new seeds.

Native budget before E32: 260 captures, 174,720 P/S episodes, 94,080 T16 episodes;
570 software-pilot episodes. E32 only replays frozen disagreement unions,
at most 161,280 episodes. One scientific risk fit, no raw fits or actor updates.
Scheduler uses unpinned node001-node006, 1 CPU/1,536 MiB native or 3,072 MiB
offline. Full samples/models stay server-side; only compact receipts/reports
are pulled. Preflight must pass before formal sampling is admitted.

Validation: 34 focused/regression tests pass; runner and submit CLI startup
pass. Checks cover honest fit/calibration ordering, forbidden test T16 access,
seed/bank separation, unchanged service proposals, exact expected thinning,
prefix-cluster calibration, duplicate-profile invariance and all-raw rejection.

Limitations: single intervention with fixed lower and finite-MC proposal,
not full end-to-end Freq-HRL or a fleet-safety theorem. Independent calibrated
scores still require untouched test control outcomes; no positive claim yet.
