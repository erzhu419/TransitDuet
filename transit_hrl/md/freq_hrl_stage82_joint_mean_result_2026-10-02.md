# Stage82 Result and Next Step

Preflight t120791/792 and full t120793-801 all exited0; code4ac1f4aacf,
20 focused tests passed.8 roots,8192 episodes /9,830,400 native steps.
Both std vectors and source/Adam/decoder frozen;128 part checks,64 joint checks,
2448 Fisher JVPs and4896 exact-KL forwards. Root compute247-257s,4.5-4.7GB peak.

Bonferroni60 equal-root bootstrap reward contrasts; joint uses.0005 KL per level,
lower-half.0005, lower-full.001 (same total conditional budget as joint):

| Period | Joint minus source | Joint minus lower-half | Joint minus lower-full |
| --- | --- | --- | --- |
| 50 | +.43717 [.20525,.61357] | +.04820 [.03297,.06253] | -.10582 [-.16878,-.02670] |
| 100 | +.84330 [.51498,1.33843] | +.07920 [.03065,.13432] | -.22556 [-.35456,-.12647] |

Joint also exceeds upper-half and joint-minus with positive corrected CIs.
Both levels contribute, but joint is inferior to lower-only at equal total budget.
Interaction is negative:50 -.00143 CI [-.00190,-.00092];
100 -.00330 CI [-.00663,-.00097]. It is small, not positive synergy.

## Limitations
Joint versus zero residual remains inconclusive:50 +.24642 CI [-.10069,.57477],
100 +.53486 CI [-.05855,1.23335]. Gains are small and teacher-initialized.
Sum-of-level source-state KL is not trajectory KL; no iterative training or
frequency-superiority claim. Stage67 critic-route HOLD and all sources stay unchanged.

Next: independent multi-update joint MC learning versus same-budget lower-only,
keeping std/decoder fixed and using preselected final evaluation, no allocation
sweep or best-checkpoint selection. Test upper's sustained value before long training.
[Compact evidence](../results/pointmaze_joint_mean_stage82_full_20261002_r1/qualification_compact.json).
