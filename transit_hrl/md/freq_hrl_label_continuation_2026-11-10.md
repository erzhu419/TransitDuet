# Paired Label Continuation

Completed: all eight probes exited 0; [readout](freq_hrl_label_continuation_result_2026-11-10.md).

Freeze the original history controller and source checks/clock policy C0. At
the same C0-visited TRAINING states, compare only R versus C0 label continuation.
Features stay 41-dimensional, ridge slope DF8, cut quantile 0.75 from training-A
leave-prefix-out predictions. B futures evaluate error; they NEVER fit/calibrate.

Reuse the original 32 fitting prefixes/root and 617 bin/839 clock draws, now on
all C0 optional checks including those not visited by R. Three states/prefix,
one new fitting A future and four independent B futures/state: 96 fitting pairs
per head. Tag1003 futures are shared across R/C0 labels. Switch continuation on
the primitive step AFTER the forced now/wait action. Verify causal state/feature
equality, shared future IDs, factual C0 replay, W+tail=G and 24-call budget.

Four native prechecks use training prefixes only. Then eight 16-prefix sampling
shards, four cache-only fit jobs and eight four-prefix probe shards, in that order.
The initial submission contains only prechecks and training sampling. All four
roots must have frozen heads before any new probe prefix is accessed. Use one CPU/2GiB per job,
dynamic unpinned placement on node001-node006; only public aggregates sync locally.
Training uses 11,212,800 native steps; later probes 4,185,600; prechecks at most 76,800.
Models and paid per-probe rows are retained on the shared server; previous
attempts are not overwritten.

Reserve eight new paths/root at root+50000..+50007, excluding registered,
effective-training and revealed seeds. In particular, root+20000 diagnostic
paths cannot calibrate this experiment. Probe C0's actual check distribution
with 4/4 tag1004 futures. Both heads are evaluated under the SAME fixed C0
continuation, with R continuation secondary. Compare train-A, independent-B,
prefix-OOF-B and fresh-prefix W error, plus paired W/tail/G gains against C0 and
between heads. Report each root, future half and conditional path interval.

No rollout uses a new head as its continuation: this tests one-step improvement,
not a newly trained closed-loop controller. A stable repair needs lower W error
and W gains without losing G across roots/halves, not just own-label fit.
Complete-episode follow-up requires a separate readout decision; it is not queued.
No threshold/encoder search, held-set selection or automatic candidate promotion.
Original HOLD, termination-competence failures and unknown old lifecycle costs remain.
Prefix OOF holds out only the new heads' labels, conditional on frozen C0;
fresh-prefix evaluation is the primary transfer test, not full-pipeline OOF.

## Execution

73 related tests passed. Native prechecks t214317-t214320 passed on node004/node006:
phase 5 causal pairing, factual replay, feature scoring and 24-call budgets held.
They used 62,400 native steps, 63.480 CPU seconds and no fits. Training shards
t214322-t214329 all completed with exit0:384 paired states,11,212,800 native
steps and9,612.543 CPU seconds, with no fits or controller updates. The
[launch receipt](../results/plan_validity_label_continuation_20261110/launch_receipt.json)
retains actual placement and costs. The readout also distinguishes live phases
from failed attempts; its targeted tests passed after this status-only correction.

Cache-only fits t214363-t214366 completed and froze both heads on every root:
96 fitting pairs/head, DF8,264 ridge calls total,8.657 CPU seconds and ZERO new
environment steps. No models or raw banks were downloaded. Only after all four
roots passed this gate were probe shards t214367-t214374 submitted. They
subsequently completed on node004/node005. The
[fit/probe receipt](../results/plan_validity_label_continuation_20261110/fit_probe_launch_receipt.json)
records the transition. The final readout does not support continuation-only
repair; no new-head closed-loop jobs followed. Frozen policies, cuts and HOLD are unchanged.
