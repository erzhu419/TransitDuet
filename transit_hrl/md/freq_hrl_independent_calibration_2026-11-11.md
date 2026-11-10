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

Four native prechecks first, at most 48,000 environment steps total. Formal work,
prepared but not released: eight calibration shards (4,224,000 steps), four zero-
simulation cut freezes, four readouts (2,112,000 steps). Total cap 6,384,000 steps;
zero full-model/ridge fits. Jobs are unpinned 1CPU/2GiB on node001-node006. Retain
paid banks server-side and retrieve only public aggregates. No automatic follow-up.

## Limitations

New cohorts are a development diagnostic on the same four trained roots, not the
original reserved confirmation cohort. These are single interventions on C0 states,
not learned-head closed-loop execution. Prefix intervals condition on trained roots;
unknown earlier lifecycle costs and original termination/confirmation HOLD remain.
