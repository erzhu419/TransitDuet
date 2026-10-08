# Stage144: Whole-Scene Cached-Credit Transfer

Stage143 compact fitting improves all four root-periods in both execution
modes, but parameter noise-panel agreement is weak/mixed. Check whole-scene
transfer before new query labels or seed expansion; no further probe sweep.

Roots410037/410049, periods50/100, horizon1200. In each of eight folds, exclude
one entire archived training scene, both A/B panels. Fit both wide/local credit
in raw392D and existing causal26D with only the other seven scenes. All feature
scaling and matched update geometry also use only those seven scenes. Keep
warm actor/lower/values/authority fixed, std0.05, RMS0.001767767/KL0.005.
Float32 RMS checking uses absolute tolerance1e-7; the update itself is unchanged.

Execute both signs on the excluded scene under both lower noise panels, mean
upper only. Record actual paired return/tracking effects separately from the
held-out linearized query gain, output RMS and functional raw/compact cosine.
No held-out label participates in fitting, sign choice, damping or selection.
Primary wide compact-plus versus raw-plus, companion local comparison; all
signs and warm comparisons retained. No winner, CI or automatic seed expansion.

Per root32replays+256held-out executions=288episodes/345,600steps, zero new
queries,32shared factorizations. Two roots576episodes/691,200steps. Sixteen
workers+parent/12GiB, dynamic node001-node006, no aggregator. Compact JSON only.
Submit all16held-out paths per period in one pool batch, not two paths per fold.

## Dispatch Revision

r1/t137094-t137095 was cancelled during execution to fix underused workers;
both terminations succeeded. No intermediate result is used. Exact partial
native cost is unknown, bounded by the planned691,200steps; retained in the r1
receipt separately from the replacement run. Algorithm, roster and budget stay
unchanged. Eight related tests passed; four reran after evaluation batching.

## Run Receipt

Implementation31bdccbcf8; preregistration16aa0d5de8 pushed before submission.
Run: `pointmaze_option_crossfit_stage144_probe_20261008_r2`.
t137097/root410037 and t137098/root410049 were queued at receipt, with no
assigned node. Both allow all six nodes,17CPUs/12GiB each, no aggregator.
Twelve server-side source files are available by metadata check only.
Preregistration and compact dispatch receipts retain the replacement lineage.

## Completed Result

t137097/node004 and t137098/node006 finished. Pulled312,225bytes of compact
JSON only. Contracts, budgets, scene exclusion, paired means and frozen checks
pass:576episodes/691,200steps, with withdrawn r1 partial cost kept separate.

| Period | Wide raw - warm | Wide compact - warm | Wide compact - raw | Local compact - raw |
| --- | ---: | ---: | ---: | ---: |
| 50 | +0.134074 | +0.390562 | +0.256488 | +0.279176 |
| 100 | +0.071761 | +0.161441 | +0.089681 | +0.026850 |

Wide compact beats warm and raw at both periods in both roots. At100,23/32
paired paths improve over raw,20/32 over warm. Local compact-minus-raw at100
is-0.001137 in root410037 and+0.054836 in root410049; do not call all probes
uniformly improved. Retain this as positive update-transfer development, not
frequency ownership or complete HRL evidence. No new task submitted here.

## Limitations

Folds share seven training scenes and are not independent seed replicates.
These are archived scenes withheld from each update fit, not untouched final
test scenes. Query gains are finite-probe diagnostics, not exact derivatives.
Output RMS is matched on training states; held-out RMS is reported, not clipped
or tuned. This addresses mean-policy update transfer, not the sampled-policy
forecast deficit, learned promotion, frequency attribution or full joint HRL.
