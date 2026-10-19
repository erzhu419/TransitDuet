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

Validation: 35 focused/regression tests pass; runner and submit CLI startup
pass. Checks cover honest fit/calibration ordering, forbidden test T16 access,
seed/bank separation, unchanged service proposals, exact expected thinning,
prefix-cluster calibration, duplicate-profile invariance and all-raw rejection.

Scheduling update (2026-10-19): `collect` submits the same 3,120 P/S and 1,680
T16 tasks together after completed captures. Scientific fitting requires both
banks completely done/exit0. Sampling, seeds, action rules and the registered
scientific contract are unchanged; E32 retains its separate model/action seal.

Execution update (2026-10-19): scientific source `1bea867609`, scheduling
revision `f33620ee36`. Freeze t183630 passed exact replay of 360 old states.
Pilots t183632-t183641 and aggregate t183655 completed done/exit0: 570 native
episodes, 80 exact identities, peak RSS 682.71 MiB; pilot labels are excluded.
Captures t183661-t183920 completed done/exit0 and passed server-only coverage
and control checks: 260 prefixes, 1,560 states, fit/calibration/test 80/60/120.
Formal collection t183945-t188744 completed all 3,120 P/S and 1,680 T16 jobs
done/exit0 (268,800 native episodes). The qualified fit t188791/node001
completed done/exit0, validating panels and saving its one ridge model before
reading calibration T16. Normal-equation error is 8.67e-19. Independent
calibration adds 0 to learned risk and 0.0652098 to the unlearned S8 score.
Training MSE is 0.000626616 on changed proposals; calibration MSE is
0.001776484 on base-eligible proposals. These are different populations,
and calibration error is not untouched test-control evidence.

The sealed test risk rule retains 77/379 baseline changed profile-state
decisions (20.3%), covering 36 physical states; bias-only retains 40. It only
retains baseline actions or returns raw, and expected thinning count error
is exactly zero. These counts are not waiting-gain retention or new seeds.
Independent E32 t188793-t189876 has 1,084 jobs over 271 disagreement states,
32,448 native episodes, with all 720 states retained in the denominator.
Actions and matching weights were sealed before dispatch; no E32 outcomes
were read for fitting or calibration. Primary joint improvement and the
50% waiting-gain retention requirement remain pending evaluation.

Qualification pulls were 4,279 pilot scalar bytes and 249 capture-audit
bytes; fitting/seal summaries added 6,369 bytes. Rows, snapshots, models,
test action arrays and thinning weights remain server-side. No raw reference
fit or actor update was added.

Limitations: single intervention with fixed lower and finite-MC proposal,
not full end-to-end Freq-HRL or a fleet-safety theorem. Independent calibrated
scores still require untouched test control outcomes; no positive claim yet.
