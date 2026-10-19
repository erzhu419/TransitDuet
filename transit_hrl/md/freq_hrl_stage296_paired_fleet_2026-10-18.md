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
were read for fitting or calibration. The primary result is reported below.

Qualification pulls were 4,279 pilot scalar bytes and 249 capture-audit
bytes; fitting/seal summaries added 6,369 bytes. Rows, snapshots, models,
test action arrays and thinning weights remain server-side. No raw reference
fit or actor update was added.

## Independent E32 result (2026-10-19)

All 1,084 tasks completed done/exit0; report t189903/node006 also completed
done/exit0. Coverage and all 32,448 episodes / 8,672 state-futures qualified.
Only the 17,348-byte scalar report was pulled. Pooled means, E16 halves and
waiting-retention arithmetic reconcile. Negative values mean improvement.

| Comparison / metric | Mean | Prefix CI95 | Conditional MC CI95 | First E16 | Last E16 |
| --- | ---: | --- | --- | ---: | ---: |
| Risk vs raw / cost | -0.000122999 | [-0.000212118, -0.000038735] | [-0.000184209, -0.000061790] | -0.000116218 | -0.000129780 |
| Risk vs raw / wait_min | -0.000331535 | [-0.000684836, -0.000100426] | [-0.000398414, -0.000264655] | -0.000323351 | -0.000339719 |
| Risk vs thinning / cost | -0.000016967 | [-0.000040269, +0.000003151] | [-0.000030895, -0.000003038] | -0.000007016 | -0.000026918 |
| Risk vs thinning / wait_min | -0.000017930 | [-0.000051529, +0.000007374] | [-0.000034017, -0.000001843] | -0.000034616 | -0.000001244 |

Relative-to-raw cost/wait improvement passes both uncertainty checks and both
halves. The preregistered primary nevertheless FAILS: waiting-gain retention
is 29.50% pooled, 27.38% / 31.85% in the halves, all below 50%. The original
zero-abstention baseline has cost -0.000172291 and wait -0.001123820; its
cost/wait intervals are also negative. It remains a comparator, not a
post-hoc replacement for the failed primary.

Filtering loses 0.000074610 service-cost benefit versus that baseline while
point fleet savings improve by 0.000025318 (both fleet intervals cross zero).
The total cost difference is +0.000049292, also inconclusive, whereas wait
degradation is supported by both intervals. Extra value versus matched
thinning has a negative MC interval and negative points in both halves, but
its prefix interval crosses zero. Neither incremental-cost nor incremental-
fleet claim passes the registered two-interval criterion.

Next: Stage297 freezes the critic and proposals, compares fleet-only versus
total-cost retention scores, and selects global cutoffs under an explicit
waiting-gain budget using fit/calibration roles only. No E32-based threshold
choice. A nontrivial development candidate must precede new-prefix confirmation.

Limitations: single intervention with fixed lower and finite-MC proposal,
not full end-to-end Freq-HRL or a fleet-safety theorem. The all-state mean
waiting benefit is about 0.020 seconds; this is not deployment-scale evidence.
Incremental learned value across prefixes and the 50% retention claim remain
unconfirmed; the failed primary is not promoted.
