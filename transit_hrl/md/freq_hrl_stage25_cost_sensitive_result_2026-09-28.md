# Stage-25 Cost-Sensitive Decision Result

**Both roots fail the frozen development gate.** Preflight `t101527` completed
on node004; full tasks `t101530/101531` completed on node004/node006 at
`9de3131c41`. Each root completed 32 supervised fits on eight held-out paths,
with no new environment steps or controller updates. Retrieved 257,148 bytes
of full-result JSON. Nineteen tests passed; independent cache recomputation
matched normalization, predictions, paired benefits and Monte Carlo SE.

Mean local ISE benefit of cost-sensitive decisions over each control:

| Root | Short-window | Uniform classification | MSE | Always wait | Always now |
|---|---:|---:|---:|---:|---:|
| 209011 | -0.013871910 | -0.015201771 | 0 | +0.003080612 | -0.009989436 |
| 209061 | 0 | 0 | +0.003626111 | -0.000883425 | +0.004283214 |

On 209011, cost-sensitive and MSE actions coincide. Relative to short-window,
four decisions change, three with negative contributions. On 209061,
cost-sensitive, short-window and uniform actions coincide. Cost weighting
therefore does not qualify the candidate; retain Stage-9, without deployment,
threshold tuning or extra seeds for this predictor.

Next: stop fitting new objectives on these 16-state caches. Redesign temporal
supervision and training-state coverage for a causal plan-validity model,
with path-disjoint evaluation frozen before collecting data. The results
do not isolate inadequate sample size as the cause.

## Limitations

These are reused-development, fixed-state comparisons under a frozen
continuation, not episode-performance or independent-generalization evidence.
