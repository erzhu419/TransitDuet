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
44 focused tests pass: original score reproduction, exact zero shift, untouched
future-trip samples, active-only shifts, model/score/budget accounting, causal
labels and staging. Full native episodes run only through scheduler.

## Limitations

Stage171 check scenes are reused for mechanism development, not independent
confirmation. The ridge corrects mean remaining time, not conditional uncertainty
or responses of future-trip durations to plans. Actual service improvements are
not implied by better prediction MAE; complete learned Freq-HRL is not tested.
