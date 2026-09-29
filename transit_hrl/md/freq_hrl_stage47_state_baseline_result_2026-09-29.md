# Stage 47: state-baseline native result

Implementation `5dec9f0dc7`; pre-outcome freeze `3c298a924b`.
Tests `t104126`:14 pass. Native `t104129`-`t104136`:eight originals exit0,
204-216s/root on node001/004/005/006. Offline `t104138` qualifies pairing,
episode-held-out fits, accounting and independent seven-endpoint statistics.

## Results

Paired-root bootstrap65536 draws, Bonferroni7; all seven effects inconclusive.

| Fixed endpoint | Mean | Adjusted CI |
|---|---:|---|
| State minus time-MC, first return | -1.23642 | [-2.57848,0.03691] |
| State minus time-MC, final return | 1.00150 | [-2.71580,5.58814] |
| State minus GAE, final return | -1.94994 | [-6.03667,2.63055] |
| State minus frozen, final return | -2.60589 | [-7.93005,0.86840] |
| Time-MC minus frozen, final return | -3.60739 | [-9.34123,0.48260] |
| GAE minus frozen, final return | -0.65595 | [-3.71941,2.29912] |
| First gradient-dispersion reduction | 0.02478 | [-0.54616,0.72815] |

First held-out prediction MSE1231.99 versus1518.63, descriptively18.88% lower,
with5/8 roots improved; score-gradient dispersion improves only4/8. Method cost:
5836800 steps/143513 upper/204772 gate calls,4864 audits,15360 actor/15360 critic
steps,36864 auxiliary fit steps/2048 score backwards; zero extra native steps.
Only the [40KB summary](../results/pointmaze_state_baseline_stage47_v1_full_20260929_r1/qualification_summary.json) is local.

## Decision And Next Step

Prediction gain does not establish variance reduction or learning utility.
Keep this variant experimental, not the default. Reused roots give conditional
development evidence; no supported repair or noninferiority. Next isolate an
**episode-scale policy-displacement budget under full-task credit**, retaining
original PPO/time-MC/frozen controls. Freeze that new intervention before running;
no seed extension, checkpoint selection or baseline retuning follows this result.
