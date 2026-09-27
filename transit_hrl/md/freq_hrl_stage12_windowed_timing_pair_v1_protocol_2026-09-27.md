# Stage-12 Windowed Timing-Pair Development Protocol

Protocol: `pointmaze_timing_pair_stage12_v1_development`

Algorithm revision: `64fe6ab72b1c466f62150284393cc54c6175a796`

Stage-11's same-budget full-episode labels were poorly predictable by the
frozen causal model. Stage-12 changes only the fitting target to paired
tracking ISE over the 50 steps after each legal check. Both arms still run
complete 1200-step episodes with the same exogenous path and one upper call
per 50-step bin. One arm plans at the check; the other waits to offset 25.
The window always includes both calls and ends before the next bin's forced
deadline. To fit the entire window, opportunities exclude the final bin;
12 bins per each of eight branch-fit paths are sampled from the remaining
noninitial bins. Paired replay remains 230,400 primitive steps per root.

Controller training, features, Ridge family, grouped path CV, 0.75 out-of-fold
threshold, held-out paths, and full-episode closed-loop evaluation match
Stage-11. Preflight is root `208001` with two pairs on each of two paths;
development reuses revealed roots `209011/209061`. Report paired-prefix and
call-budget validity, label distribution, grouped-CV error, fixed replay,
episode ISE/return, and early-call counts. Advance to a fresh-root protocol
only if Stage-12 beats **both** fixed planning and the Stage-9 candidate in
episode ISE on **both** development roots. A failed gate is retained; no
threshold tuning or sequential root extension is allowed.

Scheduler uses dynamic node001-node006 placement, one CPU and 1536 MB per
cell. Synchronize only compact `result.json`.
