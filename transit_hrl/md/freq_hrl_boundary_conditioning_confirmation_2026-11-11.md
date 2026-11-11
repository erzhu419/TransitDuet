# Frozen conditioning: fresh-prefix confirmation

The completed full-A freeze (`t214748`, exit 0) supplies both conditional
models and every unchanged control. The user now authorizes only the reserved
[confirmation plan](../protocols/plan_validity_boundary_conditioning_confirmation_2026-11-11.json),
executed through its separate
[release manifest](../protocols/plan_validity_boundary_conditioning_confirmation_release_2026-11-11.json).
The original reservation and models remain unchanged.

Budget: four roots, 16 fresh prefixes/root, three checks/prefix, four windows/check
and suffixes 40-47. Sixteen unpinned single-core scheduler shards on node001-006
use 2 GiB RAM each: 64 factual episodes, 6,144 paired futures and exactly
14,822,400 native steps. Seven frozen predictors score 5,376 windows; no fitting,
new candidate, gate revision or exposed-E reuse is admitted.

The statistical implementation is fixed before sampling: validate all sixteen
completed shard ledgers, then use 10,000 paired whole-prefix bootstrap resamples
per root. Primary Reacher1 must beat Wide, state-RMS, old eight-prefix base and
zero under the registered one-sided bounds and all suffix/window/prefix halves.
Report all roots separately. Conditional suffix MC SE is separate from prefix
uncertainty. Seventeen focused tests passed, including direct-versus-merged
moments, reproducible bounds, no MC variance double counting and rejection of
incomplete cohorts before bootstrap.

Private rows, coefficients and predictions stay on the server. Only public
progress/moment summaries are retrieved. No aggregate job waits in the queue;
analysis starts after all paid shards finish. This confirms boundary-tail
prediction, not complete-episode control or independent-training-seed gains.
The original policy recipe remains HOLD.

Launched: `t214752-t214767`, all running on scheduler-selected node002-006,
all sixteen past the global freeze gate. The launch snapshot records 468,000
native steps from six completed states, not final results. Only 3,608 public
progress bytes returned; [receipt](../results/plan_validity_boundary_conditioning_confirmation_20261111/release.json).
Implementation and statistics were pushed as `b4e853d25c` before sampling.
