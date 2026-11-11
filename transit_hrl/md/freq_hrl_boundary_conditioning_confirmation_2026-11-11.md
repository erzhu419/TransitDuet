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

## Completed Result

All sixteen tasks exited 0: 64 prefixes, 192 states, 6,144 paired futures,
14,822,400 native steps and 5,376 frozen predictor scores. No fitting or budget
extension occurred. The unchanged analyzer used 40,000 bootstrap resamples,
0.068 CPU seconds and zero native steps. All four registered root rules fail.
[Public summary](../results/plan_validity_boundary_conditioning_confirmation_20261111/summary.json)
and [completion receipt](../results/plan_validity_boundary_conditioning_confirmation_20261111/completion.json).

Tracking-gate MSE changes against each frozen reference (negative is better):

| Root | Wide | RMS | Old base | Zero |
| --- | ---: | ---: | ---: | ---: |
| PointMaze 1972110000 | +5.52% | +3.26% | +16.99% | -16.21% |
| PointMaze 1972120000 | +7.33% | +3.65% | -10.10% | -19.28% |
| Reacher 1972210000 (primary) | -0.78% | +1.70% | +0.46% | +14.33% |
| Reacher 1972220000 | +1.48% | +1.33% | +1.41% | -5.36% |

Primary paired MSE differences and nominal prefix-bootstrap bounds, in units
of 1e-4:

| Reference | Mean | Two-sided 95% | One-sided 95% upper |
| --- | ---: | --- | ---: |
| Wide | -0.491 | [-1.657, +0.464] | +0.330 |
| RMS | +1.038 | [-0.487, +3.095] | +2.653 |
| Old base | +0.282 | [-5.555, +8.237] | +6.770 |
| Zero | +7.794 | [-0.431, +19.301] | +16.912 |

The primary gain versus Wide survives both suffix/window halves but reverses
in the second prefix half; prefix 1972279000 supplies 93.43% of its net gain.
Its squared-output term falls by 4.954e-5 while alignment worsens by 4.698e-7:
the small aggregate gain is not evidence of better target alignment. Tracking
is worse than Wide on both PointMaze roots with two-sided intervals above zero.
Wide still beats zero on both PointMaze roots and Reacher2, with two-sided
intervals below zero and all half means negative. Preserve those positive results.

Decision: do not promote either gate or retrospectively select RMS. Preserve
base/Wide and the full negative readout; this exposed cohort is reporting-only.
No follow-up jobs were created. The next decision needs a distinct training-only
hypothesis, not more futures or a gate revision on this confirmation cohort.

Limits: percentages use observed MSE to the eight-future mean; inference uses
paired loss contrasts, with conditional MC uncertainty separate. This is frozen
boundary-tail prediction, not zero-action control or an episode-policy test.
