# Stage168: Isolate Waiting and Fleet Score Horizons

Stage167 reproduced all 40 frozen trajectories. Prefix score argmins disagree
with the full-tail policy in 546/880 and 564/880 decisions; 92.68%/95.57% of
tail proposals are later rewritten. Test executed policy value, not more tuning.

Use a two-by-two factorial: full/full, full-wait/prefix-fleet,
prefix-wait/full-fleet and prefix/prefix. Also reproduce window forecast.
All use the same five feasible strengths, fit-only duration model and 64 draws,
historical peak floor, native RNG, frozen lower, dispatch cap, daily endpoints
and six-trip commitments. Prefix ends at the next same-direction callback;
waiting includes the seventh departure anchor. No cost weights change.

Ten unpinned scheduler cells, roots 397/401, five regimes and all four registered
scenes. Budget: 200 frozen episodes, 12,276,000 ticks, zero updates, 80 exact
source controls and 160 selected-window trip-ledger qualifications. Primary:
actual restricted service cost versus full/full and window forecast. Report
wait/reward and fleet/nonfleet components, plus the two main effects and their
interaction. Preserve every regime/root result, including losses.

Only code and fresh run inputs are staged. Keep raw native records/checkpoints
remote; export schedules, candidate scores and compact window panels in JSON.
56 focused tests pass, including executed factorial scores, irreversible anchors,
old selector reproduction, physical contrast/interaction arithmetic, worker
ordering and scheduler staging. Run
`native_transit_commitment_scoring_stage168_frozen_20261011_r1`, revision
`c219151ef6`: t142385-t142394 are RUNNING on node001/004/005/006. Full 80-control
reproduction, 160 window-ledger qualifications and performance results are pending.

## Limitations

Local prefix waiting is a forecast proxy, not native passenger-wait credit.
Clipping fleet risk can be myopic because commitments survive past the callback.
Reused development scenes and frozen lower do not establish independent
confirmation or end-to-end learned Freq-HRL. Only an actual improvement on both
roots warrants further upper-policy learning; ranking disagreement alone does not.
