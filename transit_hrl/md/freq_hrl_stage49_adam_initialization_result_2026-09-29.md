# Stage 49: fixed-KL Adam initialization result

Version1 retired before native runs:critic-only warmup leaves actor Adam empty.
Version2 freezes checkpoint17/pre3,after the first GAE actor update,not by return.
Implementation `70c1ddc2e9`,pre-outcome freeze `9ceff8d06e`;26 tests pass.
Native `t104195`-`t104202`:all eight exit0 in509-545s/root;offline `t104893`
qualifies pairing,one-time reset/continuity,KL/costs and independent statistics.

## Results

Paired-root bootstrap65536 draws,Bonferroni9;all effects inconclusive.

| Fixed sampled-return effect | Mean | Adjusted CI |
|---|---:|---|
| Fresh MC minus inherited MC, first | 0.00506 | [-0.00159,0.01329] |
| Fresh MC minus inherited MC, final | 0.00450 | [-0.00307,0.01192] |
| Fresh GAE minus inherited GAE, final | -0.20812 | [-0.90932,0.05659] |
| Fresh MC minus fresh GAE, final | 0.17752 | [-0.32448,1.02726] |
| Credit/reset interaction, final | 0.21261 | [-0.04957,0.91342] |
| Fresh MC minus frozen, final | 0.34231 | [-0.10406,0.88964] |
| Inherited MC minus frozen, final | 0.33782 | [-0.10736,0.88421] |
| Fresh GAE minus frozen, final | 0.16479 | [-0.98027,1.06974] |
| Inherited GAE minus frozen, final | 0.37291 | [-0.12581,1.03602] |

All512 updates accept,max training-episode KL0.09989;LR scales1/2048-1/256.
Inherited Adam goes from40 to680 steps;fresh goes from empty to640,no later reset.
Execute218440 actor/218440 critic steps,retain20480/20480:10.67x nominal cost.
Trials5461,KL checks6485,native7680000 steps/183777 upper/267883 gate calls,
6400 audits,zero extra verification steps. Only the [compact summary](../results/pointmaze_adam_initialization_stage49_v2_full_20260929_r1/qualification_summary.json) is local.

## Decision And Next Step

Neither reset-repair condition nor the credit/reset interaction is supported.
Do not adopt reset. Next freeze a curvature-aware (Fisher) full-task score direction
under the same episode-KL budget,with inherited MC/GAE and frozen controls.
Test finite native utility;no reset/KL tuning or seed extension follows this result.

## Limitations

Same KL upper bound does not equal realized displacement or bound population
trajectories. Evidence is conditional development from fixed first-update sources,
not warmup16 repair or independent algorithm confirmation.
