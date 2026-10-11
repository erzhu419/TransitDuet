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

## Limitations

These reused check scenes and selected diagnostic times are development data.
Six realized continuations per state are not six independent seeds or expected
action values. A hindsight-best candidate is diagnostic, not an online oracle.
Subsequent original-policy replanning is part of each intervention's consequence;
this does not validate a corrected continuation policy or complete learned HRL.
