# Stage-13 Deployed-State Diagnostic Result

Tasks `t100988/989` completed on node004/006. Root 209011 has 64 pairs;
209061 has 61 because one path has no eligible early call and another has
only one. The 16 paths/root, seed-fixed bin selection, exact factual replay,
zero prefix differences and one-call-per-bin checks passed. Paired replay
costs were 153,600/146,400 primitive steps.

| Root | Factual class | Pairs | Mean chosen-action ISE benefit, 50 steps | Mean chosen-action benefit, full episode | Positive full-episode benefit |
|---|---|---:|---:|---:|---:|
| 209011 | Early | 32 | 0.082266 | 0.088204 | 27/32 |
| 209011 | Deadline | 32 | 0.021146 | 0.146872 | 27/32 |
| 209061 | Early | 29 | 0.083041 | 0.078276 | 26/29 |
| 209061 | Deadline | 32 | 0.006335 | 0.047703 | 18/32 |

The deployed-state diagnostic does **not** support a blanket loss of score
validity after deployment: both strata have positive mean chosen-action
benefit on both roots. Score versus 50-step benefit correlations are
0.699/0.738 across the sampled pairs. Stage-12's failed performance gate stands.

These are stratified local contrasts under a static continuation, not additive
episode improvements or independent-root confidence intervals. The next
unresolved comparison is now versus deferring **one check** under adaptive
continuation, rather than now versus a forced deadline with fixed future
planning times. No model or threshold is changed on the basis of this result.
