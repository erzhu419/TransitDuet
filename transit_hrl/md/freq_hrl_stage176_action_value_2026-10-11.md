# Stage176: Paired Native Action Values

Use the qualified Stage174/175 bidirectional data: 60 states/360 labels from
10 development scenes and two frozen lower roots. Fit physical terminal cost
differences, not forecast accuracy or proxy-score imitation. Fleet/nonfleet
heads use the original native cost weights; no fleet penalty is dropped.

Fixed ridge .1; 72 LF/physical-context by committed-action features per head.
All 15 action pairs per state are training examples, not independent samples.
Root/scene/regime IDs and outcome labels are absent from policy features.
An identity action predicts exactly zero; a zero model retains the source.

Ten leave-scene-out, five leave-regime-out and two leave-lower-root-out fits
keep entire scenes together; RMS scaling is training-only. No hyperparameter
search. A separate native screen requires negative mean held-out cost for
both roots in all three splits. One scheduler CPU task on any node001-006;
no new native episodes, no local simulation and no checkpoint downloads.
Ten focused tests pass: feature/label separation, identity, action-order
invariance, grouped fitting, physical-cost decomposition and task staging.

Task t143004 on node006 completed; compact result is 265 kB. Held-out mean
cost deltas: scene +0.016526, regime +0.013738, lower-root +0.037916.
All fail the registered native gate. Even apparent training cost is +0.000137;
this is not merely a held-out overfitting problem. Scene validation has only
491/899 concordant pairs, and fleet contributes +0.013889 of its cost delta.
Root397 has 23 positive fleet labels and no negatives; root401 has one
negative and no positives. A single fleet change costs 0.416667, much larger
than typical nonfleet differences. No native learned-policy rollout was sent.
Next collect passive preaction historical-peak and active-bus progress/age/
load/speed/holding context on these exact scenes (10 episodes, not another
420 branches); retain existing labels and require unchanged native control.

## Limitations

This screens one-action returns followed by the original greedy continuation.
It does not prove a new closed-loop policy or end-to-end learned Freq-HRL.
Only ten scenes; cross-fitting folds are dependent, not independent seed CIs.
