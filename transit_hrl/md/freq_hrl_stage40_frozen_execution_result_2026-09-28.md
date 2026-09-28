# Stage-40 Frozen-Level Training Execution Result

Tasks `t102460`-`t102499`: all 40 cells/6400 evaluation episodes complete,
exit zero, without duplicate children. Forty trajectory audits, 400 native
snapshot/mode replays, 120 probe/warmup/learning credit replays and all 16
paired warmup weights/Adam comparisons pass. Independent bootstrap passes.
Source `aa3813f47c`; pre-outcome full-matrix freeze `983d525bcc`.

| Deterministic return contrast | Mean | Eight-endpoint adjusted CI |
|---|---:|---:|
| Intrinsic-sampled vs frozen, final | -2.5048 | [-5.9105, 4.3638] |
| Intrinsic-aligned vs frozen, final | -0.7264 | [-7.0410, 5.8146] |
| Task-sampled vs frozen, final | -1.2927 | [-6.7725, 3.0998] |
| Task-aligned vs frozen, final | 1.2661 | [-2.3661, 5.0726] |
| Intrinsic alignment effect, final | 1.7785 | [-2.9639, 6.9903] |
| Task alignment effect, final | 2.5589 | [-4.2915, 11.2444] |
| Intrinsic alignment effect, first update | -0.1760 | [-3.3749, 2.7046] |
| Task alignment effect, first update | -0.9450 | [-3.3951, 0.8816] |

All eight endpoints are inconclusive. Task-aligned final mean906.4791 exceeds
frozen905.2130, but there is no supported return gain. All four learned
lower-sampled final means remain below their frozen905.8902 reference.
Final fixed-probe Gaussian KL spans0.1845-0.2743; mean action std0.2096-0.2144
versus frozen0.2146. Each learned arm charges5120 actor and10240 critic steps;
frozen0. Shared sampled warmup matches exactly within both reward pairs.

Method cost:20064000 primitive steps/467902 upper/719168 gate calls.
Verification:624000 steps/16338 upper/22501 gate calls, charged separately.
Only [the compact summary](../results/pointmaze_frozen_execution_stage40_v1_full_20260928_r1/qualification_summary.json)
is local; raw trajectories and weights remain remote.

## Limitations And Next

Eight reused roots support conditional development, not independent confirmation.
Sampled-mode means and the initial stochastic fixed probe are descriptive;
no noninferiority endpoint was registered. Upper/gate execution changes together.
Next isolate deployment-aligned critic warmup: this stage keeps shared sampled
pretraining and changes execution only during actor learning. A remaining
distribution mismatch is a hypothesis, not an established cause. Retain both
reward pairs and freeze the new contrast before outcomes; no additional seeds,
retuning or new training jobs launched.
