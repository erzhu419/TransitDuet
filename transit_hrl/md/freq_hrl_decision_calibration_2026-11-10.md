# Frozen-Score Decision Calibration

Compare the existing 0.75 OOF score quantile with one A-only decision-utility
cut, separately for BOTH frozen R-label/C-label heads on all four existing roots.
Use only the previous experiment's TRAINING cache, not its revealed probe paths.
No full models, features, labels, DF8, state coverage or simulator parameters change.

The missing per-state OOF predictions require 32 leave-prefix-out folds/head:
256 measured ridge calls total, no full-model refit and ZERO native steps.
Reproduce each saved quantile and OOF-B error; retain recovered scores server-side.

Selection uses common-C0 training-A effects for both heads. Enumerate original
cut, midpoints of distinct A-OOF scores and finite outer endpoints. Maximize
apparent G improvement relative to the original cut, subject to no apparent W
loss. Ties prefer fewer changed A actions, then proximity to the original cut,
then declared order. The original cut is feasible; no extra fallback policy.
Freeze both cuts before calculating B readouts. This is empirical A optimization,
not a safety certificate or a direct G regression head.

Read B1(first2)/B2(last2)/all4 on both OOF scores and frozen full-model scores.
Report utility-minus-quantile and both versus actual C0 actions, paired W/tail/G,
conditional MC SE, descriptive training-prefix intervals and action counts.
Allow and disclose all-now/all-wait solutions; do not call them adaptive timing.
Never choose a head or revise a cut using B. Four unpinned one-CPU/2GiB scheduler
jobs on node001-node006; only public aggregates return locally.

Limits: B contains independent future noise but was already revealed in earlier
development. These are known training prefixes, not independent-prefix policy
validation. OOF is for head scores; the pooled calibrated cut sees every A prefix,
so this is not nested prefix-OOF policy validation or full-pipeline OOF. Existing
HOLD and lifecycle/termination gaps remain; no automatic new-prefix or closed-loop run.
