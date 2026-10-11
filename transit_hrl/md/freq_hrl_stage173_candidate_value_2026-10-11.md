# Stage173: Native Single-Candidate Value

Stage172 improves global active-trip MAE but worsens actual control relative to
the original selector. Measure action value before changing another predictor
or training an upper actor on these proxy scores.

At each of ten root/regime scenes use the first original/corrected choice
divergence, or fixed zero-based index22 if none. Selection uses no reward. Replay
the original selector and each of its six candidates at that one decision,
then resume the original greedy selector. Compare exact prior action/score
prefixes, physical vehicle/passenger states, prefix cost and Python/NumPy/Torch/
scenario RNG states. Qualify the unmodified observer against Stage172, then
run the original selected candidate first and require exact full-episode identity.

Two roots, five regimes: 70 native episodes, 4,296,600 ticks, zero updates,
20 exact controls, 60 paired interventions and 70 selected-window ledgers.
Keep the frozen lower, original candidate geometry, 64 draws, fitted inputs,
score weights, fleet12 and clock61380 unchanged. Report all six actual costs,
waits/peaks, proxy ranking discordance and realized regret per common state.
Only ten unpinned scheduler tasks on node001-006, 1 CPU/3 GB each; compact JSON
returns locally, raw traces/checkpoints stay remote. Next decision follows
native action-value results, not global remaining-time prediction accuracy.
62 focused tests pass, including exact default selection, single overrides,
prefix/state/RNG mismatches, identity outcomes, continuations and accounting.

Run `native_transit_candidate_value_stage173_frozen_20261011_r1`, source revision
`ce47839ad6`, tasks `t142824-t142833`: DONE on node004/005/006. All 20 exact
controls, 60 paired interventions, 70 window ledgers and 4,296,600 ticks qualify.
Compact result JSON totals 7.75 MB; raw trajectories/checkpoints remain remote.

## Result and Next Step

Original score ranks disagree with 75/150 realized cost comparisons; corrected
ranks disagree with 83/150. Correction improves two states, worsens five, leaves
three unchanged. Original/updated mean realized regret is 0.004811/0.0239328
(root397) and 0.0093078/0.009732 (root401). Corrected-choice minus original cost
is +0.0191218/+0.0004242; wait is +0.1802/+0.0042 min.

The pool has a strictly better realized action in eight states. Nine states have
no realized peak discrimination; root401 OOD has peak14 for candidates0/1/2/5
and peak15 for candidates3/4. Fleet risk cannot simply be discarded. Root397
burst's one corrected action raises cost +0.073554 and wait +0.738 min, with
unchanged peak14. The failure persists with original continuation, isolating
a wrong initial ranking rather than requiring a corrected-policy cascade.

Next: collect native counterfactual cost labels at fixed indices8/22/36 on
separate training scenes. Learn action-value differences, not remaining-time
MAE or proxy-score imitation; retain fleet/nonfleet labels separately.

## Limitations

These reused check scenes and selected diagnostic times are development data.
Six realized continuations per state are not six independent seeds or expected
action values. A hindsight-best candidate is diagnostic, not an online oracle.
Subsequent original-policy replanning is part of each intervention's consequence;
this does not validate a corrected continuation policy or complete learned HRL.
