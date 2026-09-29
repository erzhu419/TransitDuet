# Stage 48: episode-KL native result

Pre-outcome freeze `6f12dbce8e`; Adam restore fix `fd7a9df796`.
Tests `t104168`:20 pass. Native `t104173`-`t104180`:eight originals exit0,
255-267s/root on dynamic node001/004/005/006. Offline `t104181` qualifies
exact first batches/critic/MC candidates, trial costs and six-endpoint statistics.

## Results

Paired-root bootstrap65536 draws, Bonferroni6; all effects inconclusive.

| Fixed endpoint | Mean | Adjusted CI |
|---|---:|---|
| Bounded MC minus MC, first return | 0.49968 | [-1.54081,1.96989] |
| Bounded MC minus MC, final return | 3.32956 | [-1.70999,7.12213] |
| Bounded MC minus GAE, final return | 3.43164 | [-0.74270,8.22505] |
| Bounded MC minus frozen, final return | -0.03777 | [-0.63381,0.46097] |
| MC minus frozen, final return | -3.36734 | [-7.63629,1.92193] |
| GAE minus frozen, final return | -3.46941 | [-7.97339,0.77878] |

All128 bounded updates accept; maximum deployed training-episode KL0.09836
meets the fixed0.1 budget. Selected actor LR scales1/2048-1/256;75/128 use1/1024.
Bounded trials1372. Execute65120 actor/65120 critic steps; retain15360/15360:
4.24x nominal executed optimizer cost,2396 KL checks. Method5836800 native steps,
140983 upper/203892 gate calls,4864 audits,zero extra verification steps.
Only the [59KiB summary](../results/pointmaze_episode_kl_stage48_v1_full_20260929_r1/qualification_summary.json) is local.

## Decision And Next Step

The displacement constraint works, but near-frozen behavior is not learning
repair. Neither registered final-return condition is supported. Keep experimental.
Next freeze a same-budget episode-KL comparison of inherited versus fresh Adam
state for full-task MC, with GAE/frozen controls, to test optimizer memory versus
credit-direction failure. Do not tune the KL budget or extend these roots.

## Limitations

Empirical training-history KL does not bound population trajectories or guarantee
reward. Reused roots give conditional development evidence, not independent
confirmation. No supported repair or noninferiority claim follows this result.
