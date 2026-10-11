# Stage174: Native Action-Value Training Data

Stage173 finds better realized candidates in eight of ten states, while
original/corrected proxy ranks miss 75/83 of 150 cost comparisons. Collect
matched native cost labels before learning another upper value model.

Use the first registered Stage169 scene per root/regime as training data,
disjoint from Stage173's 811-series checks. At fixed zero-based indices8/22/36,
replay an observer plus six one-candidate continuations, restoring original
greedy control afterward. Qualify exact observer/identity against Stage169;
verify the complete common causal prefix and physical/RNG state.

Collect native v2 frequency state16 (the callback alone supplies only state5),
pre-action queue/onboard/active counts, route speed, prefix cost and candidate
prefix geometry/risk. Labels are actual post-episode terminal cost differences,
separate fleet/nonfleet terms and wait differences against the unchanged action.
Physical return is -100 times cost difference, not the native episode reward.

Thirty unpinned scheduler cells, node001-006, 1 CPU/3 GB each: 210 frozen
episodes, 12,889,800 ticks, 60 exact controls, 180 paired interventions and
210 window ledgers. No learner updates; no check labels in training. Retain
raw traces/checkpoints remotely. Next: fit a fixed state-conditioned value
ranker on qualified training labels, then validate its native closed loop.

66 focused tests pass: training/check separation, native state prefix,
single-action continuations, identity controls, cost/fleet label decomposition,
fixed probe paths and scheduler inputs. Recomputing Stage173 after extracting
its continuation validator reproduces the archived summary exactly.

Run `native_transit_value_data_stage174_frozen_20261011_r1`, revision
`b88ca47df7`, tasks `t142895-t142924`: all 30 DONE on node004/005/006.
All 60 exact controls, 180 common-state interventions and 210 window ledgers
qualify. Compact JSON totals 23.26 MB; raw traces/checkpoints stayed remote.
Root397/root401 have better realized actions in 11/15 and 13/15 states, mean
best cost deltas -0.0057336/-0.0338793, with fleet discrimination in 5/1 states.
Root401's one low-noise fleet saving contributes 81.1% of its total best delta.
All probes8/22/36 are direction True: do not train a bidirectional upper from
this one-direction sample. Next collect the adjacent opposite-direction
probes9/23/37 on the same registered training scenes, then fit scene-grouped
action-value models on both directions. Value-model training remains pending.

## Limitations

Thirty states share ten training scenes and two frozen lower roots; these are
not 180 independent samples. Single-scene realized costs are noisy targets,
not expected Q values. The training scenes were used in earlier development;
this collection and any hindsight-best action are not performance confirmation.
