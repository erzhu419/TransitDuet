# Stage50: fixed-KL Fisher result

Implementation `040604d7b2`; pre-outcome freeze `0c524f05e4`. All8 native jobs
t106218-t106225 exit0. Qualification t106770 exits0; pairing, executed/retained
costs and independent root-count bootstrap pass. No protocol changes or exclusions.

Eight-endpoint Bonferroni-adjusted paired-root bootstrap,65,536 draws:

| Sampled native return difference | Mean | Adjusted CI |
|---|---:|---|
| Fisher minus MC Adam, first | -0.12647 | [-0.50080,0.02509] |
| Fisher minus MC Adam, final | 0.31938 | [-0.04232,0.71872] |
| Fisher minus Euclidean, final | 0.10538 | [-0.03099,0.29786] |
| Fisher minus frozen, final | 0.00815 | [-0.13194,0.15063] |
| Euclidean minus MC Adam, final | 0.21400 | [-0.08491,0.62462] |
| Euclidean minus frozen, final | -0.09723 | [-0.21804,0.02540] |
| MC Adam minus frozen, final | -0.31123 | [-0.64360,-0.02283] |
| GAE Adam minus frozen, final | -0.13889 | [-0.63164,0.22062] |

One negative effect, seven inconclusive. Neither registered learning-repair
condition nor curvature-specific benefit passes. Do not adopt this Fisher recipe.

All512 updates accept; maximum episode KL0.0999974. Adam LR scales1/2048-1/256;
all256 direct updates select half-scale. Execute110,080 actor/120,320 critic Adam
steps; retain10,240/20,480. Direct work:256 score gradients,1,664 Fisher-vector
products,1,280 CG iterations,512 proposals; all arms incur4,288 KL checks.
Native cost:7,680,000 steps,187,711 upper/268,821 gate calls,6,400 trace audits,
zero extra verification steps. Only the [96KB summary](../results/pointmaze_fisher_direction_stage50_full_20260930_r1/qualification_summary.json) is local.

## Next
Stop optimizer-only variations and seed extension of this failed protocol.
First establish lower-policy learnability/headroom with a native positive control,
and test whether full-task credit predicts ascent on independent rollout batches.
Use that result to choose a training objective, before resuming joint renewal learning.

## Limitations
Eight reused development roots are not independent confirmation. Equal KL caps
are not equal realized KL or compute. All128 Fisher solves stop at10 iterations;
maximum relative residual0.86378, so this is not evidence against a fully converged
natural-gradient method. Inconclusive intervals establish neither equivalence nor no-harm.
