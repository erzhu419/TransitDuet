# Stage258: Match Inner Training Futures

Compare cached prefix-excluded inner8 halves with inner16 using their union of
training futures. Freeze old keys, full actions, fit states, full basis, historical
prior/scale/RMS cap and fitting rules. Evaluate on common opposite16 futures;
report both training halves, both evaluation halves, action disagreement and
absolute per-cell full-response gaps. No new selection or held inner16 policy test.

Scheduler: one preflight then 31 fit cells and one report, single-core and unbound
on node001-node006. 640 inner16 fits / 5,120 solves; zero inner8, raw/full refits,
native episodes or actor updates. Persist new models/actions remotely for reuse;
pull scalar reports and small receipts only. Reproduce Stage257 controls.

Limitations: reused development futures and selection-conditioned strata, not
seed CI or fresh performance proof. Repeated inner16/full actions are not extra
independent evidence. Agreement with full is not ground-truth accuracy; remaining
fit-state exclusion can still change response. Stage253/248/169 stay closed.
