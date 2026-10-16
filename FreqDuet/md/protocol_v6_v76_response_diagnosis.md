# V76: frozen response-shape and extreme-label source diagnosis

Use V75's saved expanded44 original/structured predictors and cached train/now
consumed-validation labels. Reproduce the cached validation scores first.
Project each known seven-action response onto the native-anchored two-column
quadratic action basis, without fitting a deployable state model. Orthogonality
splits nonnative MSE exactly into response-shape and state-to-curve error.
Inspect actual selected actions under both components, and report projected
versus unrestricted mean32-label choices as optimistic diagnostics only.

For the frozen primary ridge, solve source contributions with its saved scaler,
training weights and ridge: per training day and fixed top 1/5/10 percent of
contexts by squared label mass. Require day contributions to reconstruct its
cached coefficients and selected predictions. This is linear label attribution,
not deleting a day and refitting; large contribution does not prove bad data.
Report longer/shorter label distributions without changing the passenger-time
objective. Keep individual contexts server-side; pull only summary JSON.

Run tests and all diagnosis through scheduler on node001-node006. Actor, dual,
main and admission gates remain frozen; no candidate fitting or simulation.
Next choose a structural repair only after separating shape failure from curve
estimation and extreme-label influence. No clipping or gate tuning from this audit.

## Outcome / Next

Source `14666fdf0e`, scheduler `t166093`, node004: exit 0, no retries;
three tests pass. Cached validation scores, the orthogonal MSE decomposition,
and exact frozen coefficient/prediction reconstruction all pass. Only the
65,685-byte summary was retrieved; no candidate fits or rollouts were run.

For expanded structured, shape accounts for 3.38% of training passenger MSE
and 2.00% of consumed-validation MSE (CV-squared 4.49%, readiness 1.54% on
validation). At its selected validation actions, predicted passenger effect
is -519.64 person-s, the label-projected effect +247.66 [86.97,431.59], and
actual +260.11 [111.46,434.48]. Of 47 harmful overrides, only one still has
a negative projected passenger effect: shape mismatch is not the main cause.

The top 1/5/10% training contexts contain 45.74/79.23/89.46% of squared
passenger label mass. Their additive contributions to the fixed validation
selection score are -86.96/-366.00/-496.16 person-s; the last is 95.48% of
the net predicted gain. Day 1750001 contributes -187.16. These are exact
fixed-scaler/ridge source contributions, not effects of deleting/refitting.
Across all training action contrasts, longer holding helps passengers in
20.01% (mean +923.65 person-s), shorter in 71.58% (mean -462.64).

Next retain the signed quadratic action basis and expected passenger-time
objective, but test a low-capacity nonlinear conditional response using current
turnover/dispatch state, with whole-demand-day cross-fitting before new label
collection. Compare against same-fold linear and action-only controls on
prediction and selected-action effects, not average MSE alone. Do not clip or
remove high-impact contexts, promote the label-selected quadratic choices, or
start another actor-training sweep. Actor/dual/main and admission gates stay
frozen. Limitations: projected labels use the same mean32 outcomes and their
choices are optimistic; the twelve V75 validation days are now development
evidence, and linear source concentration does not establish corrupt labels.
