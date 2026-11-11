# Nested Boundary Credit Precheck

The E8 scalar-cut branch remains closed. Reuse only the four frozen C0 controllers
and timing rules, not E8 prefixes or labels. Earlier paired-value fits and
[direct G targets](freq_hrl_target_credit_result_2026-11-10.md) did not transfer
reliably. [Stage-18](freq_hrl_stage18_state_noise_protocol_2026-09-27.md) resampled
tails after one fixed window; this design also samples independent realized windows.

At decision t, share a window seed between now/wait. At b=t+50, hold each arm's
realized window fixed and redraw the suffix, sharing a new suffix seed between
arms. Record causal pre-action boundary features, clock, remaining horizon and
the already-spent call indicator. Boundary context is label-side only. Each pair
reconstructs G = W + tail. Two-level moments separate variation across window
realizations from residual suffix variation; finite-sample negative corrections
are retained. Suffix siblings are not independent window draws.

Four mechanical prechecks only: one fresh prefix/root, first reachable nonzero
optional phase, two windows with two suffixes each. Per root: one factual episode,
two original arms, eight nested arms, 13,200 steps. Total 52,800 steps; no fits or
optimizer updates. Each arm replays the entire episode and pays its full cost;
this is not a lower-variance estimator claim at matched full-replay cost. Seeds
are checked against registered, historical and effective training roles before
submission. Only public precheck summaries return locally.

Passing checks establish the data path, not a tail-value or performance gain.
Next design the budgeted variance readout before training any boundary critic.
An action-sensitive tail target must beat zero-tail on whole held-out prefixes
before policy testing. No formal sampling, training, E8 retuning or original
confirmation is released by this protocol. `PILOT_RECIPE_HOLD` remains.

## Execution

All four prechecks t214553-t214556 exited 0 on node002/node003/node005 at
implementation `2b850b5950`. Factual C0 replay, fixed-window and boundary-context
invariance, and W+tail reconstruction passed on both tasks. Total: 52,800
steps, 55.625 measured CPU seconds, 10,795 public bytes and zero fits. The 52
related tests passed; a focused rerun also checks both native driver fork calls
and their shared seeds in the deterministic test plant. See the
[execution summary](../results/plan_validity_boundary_credit_precheck_20261111/summary.json).
No formal variance experiment or model training was submitted.
