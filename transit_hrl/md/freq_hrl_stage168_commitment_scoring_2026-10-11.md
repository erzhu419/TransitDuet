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
`c219151ef6`: t142385-t142394 are DONE. All 200 episodes, 80 exact source
reproductions and 160 window-ledger qualifications pass.

| Scoring | Cost vs window, root 397/401 | Cost vs full/full, root 397/401 |
| --- | --- | --- |
| Full wait / prefix fleet | +0.127522 / +0.069565 | +0.079172 / -0.018943 |
| Prefix wait / full fleet | +0.037369 / +0.048339 | -0.010981 / -0.040169 |
| Prefix wait / prefix fleet | +0.159113 / +0.063731 | +0.110762 / -0.024777 |

Lower cost is better. Prefix-wait/full-fleet also improves reward over full/full
by +9.033/+11.559, but loses to window forecast by -4.022/-1.741. Its wait
delta versus window is -0.3167/-0.2027 min; extra fleet cost +0.0625/+0.0625
exceeds nonfleet savings -0.025131/-0.014161. Both roots improve cost in
hour-burst, but persistent shift worsens relative to full/full: +0.167476/+0.185953.

Post-hoc coverage diagnosis: at all 40 initial direction decisions per root,
none of the five rolling candidates has the window forecast's first-six
planned departures. Stage169 adds that fixed-window primitive, preserves the
hybrid score, and requires a forced-incumbent native reproduction before testing
the augmented pool. No training expansion follows from this negative frontier.

## Limitations

Local prefix waiting is a forecast proxy, not native passenger-wait credit.
Clipping fleet risk can be myopic because commitments survive past the callback.
Reused development scenes and frozen lower do not establish independent
confirmation or end-to-end learned Freq-HRL. Only an actual improvement on both
roots warrants further upper-policy learning; ranking disagreement alone does not.
