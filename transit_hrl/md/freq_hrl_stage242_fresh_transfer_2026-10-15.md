# Stage242: fresh-future prefix and refit transfer

Stage241 found that the fixed joint/raw/half-safe cost gain flips sign across
independent future halves, while all-MSE/original/pooled has a positive secondary
pooled result. Neither observation alone identifies held-prefix transfer loss.

## Protocol

- Run: `native_transit_fresh_transfer_stage242_development_20261015_r1`.
- Reuse all Stage241 futures and Stage237/239/240 cached models and selections.
- Five endpoints: cached54/2, residual60/2 with fixed54 reference, native60/2,
  residual60/4 with fixed60/2 reference, and native60/4.
- Compare seen54, OOF probe6, full train60 and outer held60 states per model.
- Compare old inner2, old outer4, old matched controls, fresh halves4+4, fresh8.
- Reproduce all five cached Stage240 probe/train/held endpoint metrics and all
  80,640 Stage241 probe action records before interpreting fresh transfer.
- Write all action vectors before reading fresh panels; no future-driven choices.
- Separate fleet/nonfleet cost and mean-label/draw-level wait harm; own/common raw.
- Preserve all 21 candidates; focal primary joint/raw/half-safe plus raw control.
- All-MSE/original/pooled is an explicitly post-Stage241 exploratory secondary.
- One compressed panel preparation, 32 evaluation tasks and one report task.
- Zero new raw/residual/RL fits and zero new native episodes.
- Full arrays/actions/metrics stay remote; only a reduced aggregate is synced.
- Scheduler node001-node006, unpinned, one CPU and 3 GB RAM per task.

## Status And Next Step

Seven focused tests passed, including cached endpoint reproduction, fixed actions
under poisoned future labels, panel axes, physical cost units and weighted aggregation.
Preparation t155550 completed with exit zero: 240 states, 40 prefixes, a 218,484-byte
server-only old/fresh cache. Evaluation tasks t155558-t155589 are dispatched, eight
each on node001/004/005/006. Aggregate only after all32 finish with exit zero.
Use seen-versus-held and fixed-reference-versus-native differences to choose
the next training change; do not adopt the secondary just from Stage241 means.

## Limitations

Reused development prefixes; Stage241 outcomes were already inspected. This is
a diagnostic, not fresh confirmation, independent-seed CI or complete learned HRL.
Reserved Stage169 confirmation cohorts and deployment defaults remain unchanged.
