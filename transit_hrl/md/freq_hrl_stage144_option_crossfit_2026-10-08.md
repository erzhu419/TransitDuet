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

## Limitations

Folds share seven training scenes and are not independent seed replicates.
These are archived scenes withheld from each update fit, not untouched final
test scenes. Query gains are finite-probe diagnostics, not exact derivatives.
Output RMS is matched on training states; held-out RMS is reported, not clipped
or tuned. This addresses mean-policy update transfer, not the sampled-policy
forecast deficit, learned promotion, frequency attribution or full joint HRL.
