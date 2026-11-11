# Independent-Prefix Cut Calibration

Keep all eight full heads, their original numeric quantile cuts, the previous
full-score cuts, controller and C0 prefix/continuation policy unchanged. New
precheck, calibration and readout cohorts are disjoint from each other and all
exposed/reserved plan-validity seeds, including effective controller-training seeds.

Per root: 16 new calibration prefixes, three causal checks/prefix, A4+B4 futures.
Use ONLY A4 and actual full-model scores to maximize G subject to W not below
the original quantile policy; retain existing candidate cuts/ties. Freeze both
heads' cuts before calibration B and all four roots before accessing readout paths.
No winner selection, B-based retuning, coefficient fitting or controller updates.

Read eight different new prefixes/root, three checks/prefix and E8 futures. Report
E4/E4/all8 gains against quantile, previous full-score calibration and actual C0,
paired W/G, conditional MC SE, prefix t95 intervals and action counts. Keep roots
and both heads separate. Positive point behavior requires both W/G in both halves
and both roots/task; intervals and C0 comparisons must also be shown. A gain against
quantile is not automatically a gain against C0 or an episode-performance claim.

Four native prechecks first, at most 48,000 environment steps total. Formal work
is released by stage: eight calibration shards (4,224,000 steps), four zero-
simulation cut freezes, four readouts (2,112,000 steps). Total cap 6,384,000 steps;
zero full-model/ridge fits. Jobs are unpinned 1CPU/2GiB on node001-node006. Retain
paid banks server-side and retrieve only public aggregates. No automatic follow-up.

## Limitations

New cohorts are a development diagnostic on the same four trained roots, not the
original reserved confirmation cohort. These are single interventions on C0 states,
not learned-head closed-loop execution. Prefix intervals condition on trained roots;
unknown earlier lifecycle costs and original termination/confirmation HOLD remain.

## Precheck Execution

All four prechecks t214471-t214474 passed on node004/node006: 8,400 native steps
each (33,600 total), phase5 checks and 24 factual upper calls. Factual replay,
conditional pairing and both heads' native/full-score equality passed; no model
fits or updates. Measured phase CPU: 34.354 seconds. Retrieved 10,585 public bytes,
no raw banks/models. Forty related tests passed. See the
[receipt](../results/plan_validity_independent_calibration_20261111/precheck_summary.json).
At precheck completion, formal sampling was unsubmitted. The next release is only
the eight frozen calibration shards, then cut freezing before fresh-prefix readout.

## Calibration Sampling Release

The subsequent authorized step submits only the eight frozen calibration shards,
t214476-t214483, with 528,000 steps/shard (4,224,000 total). Each shard contains
eight new prefixes, three states/prefix and A4+B4 futures/state. No parameters,
cohort rosters, models or selection rules change. At sampling release, cut-freeze
and fresh-prefix readout jobs were unsubmitted. The dated
[release receipt](../results/plan_validity_independent_calibration_20261111/sampling_release.json)
records placement/status without changing the original frozen manifest.
All eight jobs started on node004/node005 (four each). The initial progress snapshot
showed at least one completed prefix/shard without errors. No raw banks or models
were retrieved.

## Calibration Completion

All eight samples completed at exactly 4,224,000 steps. The subsequent four
zero-simulation A4-only jobs t214490-t214493 froze both heads/root before B4.
The [result and next decision](freq_hrl_independent_calibration_result_2026-11-11.md)
retain conditional positives, G half-group instability and sampled decision
collapse. At that analysis, all cuts were frozen and the recommended E8 readout
was unsubmitted; no original confirmation or episode follow-up was released.

## Frozen E8 Release

The subsequent user-authorized step releases all four declared readout jobs,
t214504-t214507: 528,000 steps/root, 2,112,000 total. The 32 disjoint new prefixes
provide 96 states and 768 paired conditional futures. Both heads, all eight cuts,
models, comparators and E4/E4 split remain fixed; no further calibration is scheduled.
The [release receipt](../results/plan_validity_independent_calibration_20261111/readout_release.json)
records launch status separately from completed evaluation. This is the planned
single-intervention prefix-transfer diagnostic, not original-cohort confirmation
or a learned-head complete-episode experiment.
All four jobs started: three on node004 and one on node005. Initial public
progress confirmed at least two complete prefixes/root without errors, before
any outcome was read. No raw banks or models were retrieved.

## E8 Completion

All four readouts exited 0 at the exact budget. The
[E8 result](freq_hrl_independent_calibration_result_2026-11-11.md#frozen-e8-result)
retains a local PointMaze root2 signal, but no head meets the both-root joint W/G
criterion. Scalar-cut repair stops here; all cuts and paid results remain frozen.
No follow-up jobs or original confirmation are released.
