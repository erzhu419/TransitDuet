# Stage245: feasible-action MSE and regret ranking

Run `native_transit_feasible_critic_stage245_development_20261015_r1`: all33 tasks
done/0, no retries (t155725, t155729-t155759, t155762);15 tests passed.
All39,600 frozen Stage244 control fields reproduce exactly. Zero raw-reference
fits/native episodes. Only the1,083,979-byte aggregate was manually pulled.

## Design And Results

Fix B-target/B-calibration, raw references,198 features,17 candidates, wait gate
and A-MSE RMS cap. Optimize the same k-dimensional cost sum with cached head
difference frozen; new sum corrections are split equally, not independent head
refits. Use A's radius explicitly, not the smaller B/B initialization norm.
Primary: `rank_regret_pooled_safe`; unchanged OOF selection. Mixed Transit
full-held/fresh8 results versus raw, negative better:

| Pooled-Safe Objective | Cost Delta | Wait Delta (min) | Changed States |
|---|---:|---:|---:|
| Frozen All-MSE GLS | -0.000129902 | -0.000213411 | 6 |
| Feasible MSE | -0.000095842 | -0.000419531 | 7 |
| Rank/Regret (primary) | +0.000039061 | -0.000127214 | 7 |

Feasible MSE improves pooled cost/wait in both fresh halves, but root401 still
harms both (+0.000007957/+0.000060156). Wait is2.68 conditional MC SE from zero;
cost only1.47 SE. Ranking improves root401 wait in both halves but adds fleet
cost; root397 cost/wait both worsen. Train-prefix fresh wait also fails to improve.
All128 full ranking fits lower loss and meet KKT;125 hit the fixed RMS cap.
Of7 changed ranking states,4 have positive cost deltas on both old A/B despite
negative model margins. At401/914010103/36, fleet difference is0/8 old draws but
nonzero1/8 fresh draws: separate sparse-tail sampling from cost-sign misprediction.

## Next And Limits

Retain feasible MSE as a development candidate, not a new default. Next compare
variance-weighted paired-margin regression with gap-only ranking at fixed capacity;
diagnose service sign errors separately from fleet-tail sampling. No blind cap tuning.
Reused development states/futures, conditional SE not seed CI; one-upper-action
frozen-suffix probe, not end-to-end HRL. Defaults/reserves unchanged; models remote.
