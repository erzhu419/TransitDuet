# Stage-19 Cached-Label Averaging Result

Tasks `t101361/101362` completed on node004/node006. Run:
`pointmaze_averaged_label_stage19_v1_development_20260927_r1`, submitted at
`a8fbaa34bc`. Retrieved 48,630 bytes. Each root used 16 cached opportunities,
128 future contrasts and 16 path-held-out fits/1,024 updates. Matched states,
initialization, normalization, label joins and metrics passed independent
recomputation. New environment steps and controller training: zero.

| Root | Corrected zero MSE | Single-draw MSE | Averaged-label MSE |
|---|---:|---:|---:|
| 209011 | 0.002319738 | 0.002294605 | 0.002254474 |
| 209061 | 0.000945674 | 0.023586031 | 0.053225939 |

All columns subtract the same query-mean sampling correction. **The two-root
gate failed.** Averaging improves on single labels by 1.75% on 209011 but
worsens MSE by 125.67% on 209061; only two of eight path means improve on
each root. No deployment or fresh-root extension follows.

Post-result inspection found large cross-path extrapolation, not a near-zero
scale artifact: at root 209061/path 2184201/check 365, the averaged prediction
is -0.857762 versus repeat mean 0.004737. This path reaches 11.1 training
standard deviations in waypoint_error_1 (training scale 0.04039). This is a
diagnostic association, not a proven cause. Averaging alone does not qualify
this small-data critic; next test a lower-capacity, regularized paired predictor
on the same cached states before spending more on repeats or controller replay.
