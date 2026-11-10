# Frozen Prefix And Continuation Diagnostic

Fix original history (R) and the complete checks/clock repair (C), on the same
saved upper/lower controller. Keep both tasks and all four trained roots; C is
chosen uniformly as the mechanism under diagnosis, not as a per-task winner.
No fitting, gate changes, controller updates or candidate promotion occur.

Cross prefix R/C with continuation R/C, then force now/wait-one-check at the
same available clock. Switch continuation on the NEXT primitive step, preserving
the causal state and the forced action. Same-policy factual arms must reproduce
the original episode, and all four continuations/futures must preserve each
prefix. Every complete execution still has 24 upper calls. Check W+tail=G.

Reserve eight new paths/root at root+20000..+20007, excluding current registered,
exposed and resolved primary-training seeds. Sample three bins/path (tag960),
then one uniformly chosen clock (tag961) actually optional under BOTH policies.
Use eight identical conditional-future seeds per paired clock across all cells
(tag962), split4/4; do not choose actions using those futures. Record candidate
W error, W/G disagreement and frozen action gains under each continuation.

Compare continuation effects at fixed prefix, prefix effects under fixed
continuation, and their interaction. Compute paired-future MC standard errors
before aggregating probes; group by path for descriptive within-root CIs. Shards
and futures never increase the number of trained roots. Report A/B/all panels.
Also count exclusive optional clocks and fixed-grid state occupancy relative to
the original fitting bank. Common support is not the entire policy state space;
prefix effects compare different physical states at matched exogenous clocks.
These diagnostics are not a decomposition theorem for total policy return.

Four native prechecks reproduce both frozen sources on their saved episodes,
scan the first four TRAINING prefixes for the first shared later check, and
verify same/cross continuation replay with two futures. After all pass, run two
four-path shards/root: eight one-CPU/2GiB jobs, dynamically placed on node001-node006.
Each root uses at most 2,092,800 main simulator steps under this fixed design;
eight shards total 8,371,200. Actual failed/replayed costs remain charged.
Save paid probes incrementally on the server; retrieve only compact public JSON.
Parent HOLD, failed termination competence and unknown old lifecycle costs remain.
