# Stage-18 State and Future-Noise Result

Tasks `t101288/101289` completed on node004/node006 at submission revision
`503ac8e0aa`. Run: `pointmaze_state_noise_stage18_v1_development_20260927_r1`.
Retrieved two result JSONs totaling 107,146 bytes; no histories or checkpoints.
Each root completed 96 pairs, 16 path-held-out critic fits/1,024 updates,
16 noise opportunities with eight paired futures each, and 547,200 new replay
steps, additional to controller reconstruction and inherited supervision.
Sample counts, path separation, matched initialization, 424 inputs/31,425
parameters and reported MSE/noise summaries were independently checked.

| Root | Zero MSE | Compact MSE | Full-history MSE |
|---|---:|---:|---:|
| 209011 | 0.002280245 | 0.002477249 | 0.002623268 |
| 209061 | 0.002958750 | 0.003922665 | 0.006137094 |

**History qualification failed on both roots.** Full history is 5.89%/56.45%
worse than the matched compact critic and 15.04%/107.42% worse than zero.
Conditional within-state variance is 0.001492032/0.002098787; corrected
between-state mean variance is 0.002477943/0.000889557. Corrected conditional-
mean MSE (zero/compact/history) is 0.002319738/0.002077419/0.002440105 on
209011 and 0.000945674/0.001849424/0.004126335 on 209061.

Future variability is present, but the current full-history critic also loses
to zero against repeated-future means. This does not support a noise-only
explanation. Estimates cover 16 states/root conditional on simulator latent
state; they neither prove observable-state sufficiency nor policy improvement.
No deployment, tuning or root expansion is admitted; Stage-9 remains the
validated same-task reference. Next: use cached repeats for a matched-state,
matched-network single-draw versus averaged-label diagnostic with whole-path
holdout, without additional environment replay or controller training.
