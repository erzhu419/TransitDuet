# Stage-41 Critic Warmup Execution Alignment Result

Tasks `t102593`-`t102632`: all40 original cells/6400 evaluation episodes
complete without duplicate children. Full native audit `t103509` exits zero:
40 trajectory audits,400 snapshot/mode replays,120 probe/warmup/learning
credit replays and16 warmup/first-learning pair comparisons pass.
Independent eight-endpoint bootstrap passes. Implementation `e72fe5e468`;
pre-outcome full-matrix freeze `1543cb8c9d`.

| Deterministic return contrast | Mean | Eight-endpoint adjusted CI |
|---|---:|---:|
| Intrinsic-sampled vs frozen, final | -2.6314 | [-6.0151, 0.6558] |
| Intrinsic-aligned vs frozen, final | -3.3604 | [-9.4481, 1.9462] |
| Task-sampled vs frozen, final | -2.0294 | [-4.7072, 1.1572] |
| Task-aligned vs frozen, final | -0.9899 | [-3.0119, 0.7976] |
| Intrinsic warmup-alignment effect, final | -0.7290 | [-6.1743, 4.3353] |
| Task warmup-alignment effect, final | 1.0395 | [-1.2319, 3.9083] |
| Intrinsic warmup-alignment effect, first update | 1.2045 | [-0.1545, 2.5518] |
| Task warmup-alignment effect, first update | 0.3676 | [-0.3393, 1.1088] |

All eight effects are inconclusive. All four learned final means are below
frozen in deterministic and lower-sampled deployment. Frozen returns are
910.1372/908.7065; task-aligned returns909.1473/908.2283, respectively.
Paired post-warmup actors/frozen values and first native learning episode
transitions match exactly. Mean lower-critic parameter distances are3.7453
intrinsic/9.7912 task; first-learning old-value RMSE0.0676/7.6230.
Each learned arm charges5120 actor/10240 critic optimizer steps; frozen0.

Method cost:20064000 primitive steps/490877 upper/710476 gate calls.
Verification:624000 steps/18390 upper/23050 gate calls, charged separately.
Only [the compact summary](../results/pointmaze_warmup_alignment_stage41_v1_full_20260928_r1/qualification_summary.json)
is local; raw trajectories and weights remain remote.

## Limitations And Next

Eight reused roots support conditional development, not independent confirmation.
Lower-sampled means are descriptive; no noninferiority endpoint was registered.
Upper/gate warmup execution changes together; task reward also removes the
intrinsic action penalty. Native audit success does not establish utility.
Warmup alignment has no supported repair. Next isolate critic-only option-age
and remaining-horizon context: gate sees both explicit clocks, lower reward
critic lacks them. Keep actor inputs, PPO, rewards and budgets matched.
The information gap is a hypothesis, not an established cause. No additional
seeds, retuning or new training launched.
