# Stage207: frozen wait fitting versus transfer

Reuse94 Stage206 GLMs and47 folds. Compare training mean8 against held-out
mean8, keeping single-draw training separate. Scene comparisons additionally
match the training root/regime. Report expected-wait errors, own-loss boundary
errors, intentional lambda2 rejections and zero-signal actions separately;
include pointwise-oracle loss excess and repeated errors in both future halves.
Decompose actual selected-action wait into harm and benefit contributions.
Repeated state/model evaluations are not additional independent states/seeds.

Run: `native_transit_wait_transfer_stage207_diagnostic_20261013_r1`.
One unbound scheduler task,node001-node006,1CPU/3072MB; no fitting,optimizer
updates or native episodes. Models,bank,targets and draw panels stay remote.

Results pending. Next: choose a representation/coverage change only after
separating training-fit errors, held-out degradation and intentional rejection;
keep cost/menu frozen and preserve the unsuccessful primary's claim boundary.

Limits: in-sample diagnostics are optimistic and mean8 targets remain noisy.
Loss excess includes regularization and model restrictions; it cannot alone
prove a missing causal feature. No threshold tuning, CI or deployment claim.
