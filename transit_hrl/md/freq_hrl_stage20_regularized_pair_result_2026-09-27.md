# Stage-20 Regularized Paired Predictor Result

Tasks `t101372/101373` completed on node006/node005, submitted at `1c04d532f9`
as `pointmaze_regularized_pair_stage20_v1_development_20260927_r1`. Retrieved
209,644 bytes. Each root used 16 cached opportunities, 128 future contrasts
and 16 closed-form fits. New environment steps, controller training and gradient
updates: zero. Cache joins, path isolation, normalization, ridge normal equations,
stored-coefficient predictions and MSE were independently verified.

| Root | Corrected zero MSE | Stage-19 mean MSE | Ridge mean MSE |
|---|---:|---:|---:|
| 209011 | 0.002319738 | 0.002254474 | 0.001737412 |
| 209061 | 0.000945674 | 0.053225939 | 0.000962561 |

**The two-root gate failed.** Ridge reduces error versus Stage-19 by
22.93%/98.19%, but relative to zero improves 25.10% on 209011 and worsens
1.79% on 209061. Mean labels beat single labels by 21.96%/41.43% within ridge.
Five of eight path means improve over zero on 209011, versus two on 209061.
Mean effective degrees of freedom are 1.648/1.610; largest absolute predictions
are 0.07151/0.02779. The large neural prediction errors are substantially reduced
on these cached states, without establishing a qualified critic or policy gain.

Preserve the failure and freeze coefficients, alpha, features and roots. Next
proposed diagnostic: score these fixed out-of-path predictors using independent
conditional-future draws at the same states, with a separately frozen budget
and comparison. This can test label precision, not cross-state generalization;
no retuning, deployment or fresh-root expansion is admitted by this result.
