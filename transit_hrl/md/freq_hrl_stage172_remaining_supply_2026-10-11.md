# Stage172: Remaining-Time Correction in Native Candidate Selection

Stage171 improves remaining-time MAE on both roots and all ten fresh check
regimes. Test whether this becomes actual control value, keeping the six-candidate
geometry, scoring horizons/weights and physical objective unchanged.

Four conditions on every Stage171 check scene: window source, original augmented
selector, zero-correction selector and fit-only corrected selector. Add the
registered Stage171 additive correction to each active-trip residual member, then
apply the existing one-second floor. Future-trip draws, duration EWMA and history
floor stay unchanged. No joint copula, refitting or learned policy updates.
Use exact source-window reproduction and exact original/zero-correction physics,
schedules, candidate scores and window ledgers to isolate the integration.

Two roots, five regimes, one scene each: 40 frozen native episodes, 2,455,200 ticks,
20 exact controls and 30 selected-window ledger qualifications. Compare actual
cost/wait/reward and fleet/nonfleet terms against both window and original selector.
Record causal active corrections and post-episode remaining-time diagnostics;
do not change scores based on those labels. Ten unpinned scheduler tasks,
node001-006, 1 CPU/3 GB each. Stage only code and fresh model inputs.
45 focused tests pass: original score reproduction, exact zero shift, untouched
future-trip samples, active-only shifts, model/score/budget accounting, causal
labels and staging. Full native episodes run only through scheduler.

## Result and Next Step

Run `native_transit_remaining_supply_stage172_frozen_20261011_r1`, tasks
`t142788-t142797`: DONE on node004/005/006, unpinned within node001-006.
Source revision `bf68f2f463`; 20 exact controls, 30 window ledgers and all
2,455,200 native ticks qualify. Sync compact JSON only; no raw logs or checkpoints.

On the corrected trajectories, active sample-mean MAE falls from 98.00 to 87.30 s
(root397) and 98.10 to 91.68 s (root401). Yet corrected minus original selector
mean cost is +0.0195042 / +0.0008626; wait is +0.1824 / +0.0044 min; reward is
-9.2616 / -6.5646. All ten actual peaks are unchanged. Against window, mean cost
is -0.0134118 / +0.1009998: not a consistent improvement.

51 of 440 indexed choices differ, across seven scenes. At all seven first
divergences candidate prefixes and wait scores match; fleet-score changes flip
the ranking. Root397 burst contributes +0.074788 cost / +0.747 min wait;
its first-divergence active MAE worsens from 44.64 to 68.10 s despite better global
MAE. Full-episode differences include subsequent replanning, not a causal value
estimate for an isolated action. Reject this correction as a performance upgrade.
Next: qualify candidate fleet-risk ranking and counterfactual action value at
common native states before fitting another upper policy; do not expand seeds
or tune this mean correction on the reused checks.

## Limitations

Stage171 check scenes are reused for mechanism development, not independent
confirmation. The ridge corrects mean remaining time, not conditional uncertainty
or responses of future-trip durations to plans. Actual service improvements are
not implied by better prediction MAE; complete learned Freq-HRL is not tested.
